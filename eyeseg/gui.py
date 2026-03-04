import sys
import cv2
import pandas as pd
import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPen, QBrush, QColor, QImage, QPixmap, QPainter
import pyqtgraph as pg

class SessionModel(QtCore.QObject):

    frame_changed = QtCore.pyqtSignal(int)
    labels_changed = QtCore.pyqtSignal()

    def __init__(self, video_path, tracking_csv):
        super().__init__()

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self._current_frame = 0
        self._last_read_frame = -1
        self._cached_frame = None

        self.tracking = pd.read_csv(tracking_csv, header=[0,1,2])
        self.labels = pd.DataFrame(columns=["start", "end", "category"])


    @property
    def current_frame(self):
        return self._current_frame

    def set_frame(self, idx):
        idx = int(np.clip(idx, 0, self.total_frames - 1))
        if idx == self._current_frame:
            return
        self._current_frame = idx
        self.frame_changed.emit(idx)

    def get_frame(self):

        if self._current_frame == self._last_read_frame + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
            ret, frame = self.cap.read()

        if not ret:
            return None

        self._last_read_frame = self._current_frame
        self._cached_frame = frame
        return frame

    def add_label(self, start, end, category):
        new_row = {"start": start, "end": end, "category": category}
        self.labels = pd.concat(
            [self.labels, pd.DataFrame([new_row])],
            ignore_index=True,
        )
        self.labels_changed.emit()

    def delete_label(self, index):
        self.labels = self.labels.drop(index).reset_index(drop=True)
        self.labels_changed.emit()

    def save_labels(self, path):
        self.labels.to_csv(path, index=False)


DIVERGING_4 = (
    (178, 24, 43),    # strong red
    (239, 138, 98),   # light red
    (33, 102, 172),   # strong blue
    (103, 169, 207),  # light blue
)

class VideoWidget(QtWidgets.QGraphicsView):

    def __init__(self, model):
        super().__init__()
        self.model = model

        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)

        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.keypoint_items = []
        self.overlay_visible = True

        self.model.frame_changed.connect(self.update_frame)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)

        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def update_frame(self, frame_idx):
        frame = self.model.get_frame()
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.pixmap_item.setPixmap(pixmap)

        self.scene.setSceneRect(0, 0, w, h)
        self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.update_overlay(frame_idx)

    def update_overlay(
            self, 
            frame_idx, 
            r = 2,
            colors = DIVERGING_4       
        ):

        # Clear old keypoints
        for item in self.keypoint_items:
            self.scene.removeItem(item)
        self.keypoint_items.clear()

        if not self.overlay_visible:
            return

        row = self.model.tracking.heatmap_tracker.iloc[frame_idx]
        count = 0
        for side in ['left', 'right']:
            points = []
            for position in ['front', 'back']:
                x = row[f"eye_{side}_{position}"].x
                y = row[f"eye_{side}_{position}"].y
                points.append((x,y))
                
                ellipse = self.scene.addEllipse(
                    x - r, y - r, 2*r, 2*r,
                    pen=QPen(QColor(*colors[count])),
                    brush=QBrush(QColor(*colors[count]))
                )
                ellipse.setZValue(2)
                self.keypoint_items.append(ellipse)
                count += 1

            line = self.scene.addLine(
                points[0][0],points[0][1],points[1][0],points[1][1],
                pen=QPen(QColor(*colors[count-1]))
            )
            line.setZValue(1)
            self.keypoint_items.append(line)

    def toggle_overlay_visibility(self):
        self.overlay_visible = not self.overlay_visible
        self.update_overlay(self.model.current_frame)
        
def get_eye_angles_from_keypoints(tracking: pd.DataFrame):

    def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray):
        cos_angle = np.sum(v1 * v2, axis=1)
        sin_angle = np.cross(v1, v2)
        angle = np.arctan2(sin_angle, cos_angle)
        return angle

    left_front = tracking.heatmap_tracker.eye_left_front[['x', 'y']].to_numpy()
    left_back = tracking.heatmap_tracker.eye_left_back[['x', 'y']].to_numpy()
    right_front = tracking.heatmap_tracker.eye_right_front[['x', 'y']].to_numpy()
    right_back = tracking.heatmap_tracker.eye_right_back[['x', 'y']].to_numpy()

    # origin top-left
    left_vector = left_back - left_front  
    right_vector = right_back - right_front

    left = compute_angle_between_vectors(left_vector, np.array([0,1]))
    right = compute_angle_between_vectors(right_vector, np.array([0,1]))

    return np.rad2deg(left), np.rad2deg(right)

class TimeSeriesWidget(pg.PlotWidget):

    def __init__(self, model, window_seconds=5.0):
        super().__init__()
        self.model = model
        self.window_seconds = float(window_seconds)

        self.left, self.right = get_eye_angles_from_keypoints(self.model.tracking)
        n = len(self.left)
        self.time = np.arange(n) / self.model.fps

        self.left_curve = self.plot(self.time, self.left, pen=pg.mkPen('b'))
        self.right_curve = self.plot(self.time, self.right, pen=pg.mkPen('g'))
        self.frame_line = pg.InfiniteLine(angle=90, movable=False)
        self.addItem(self.frame_line)

        self.region_items = []

        self.setDownsampling(auto=True)
        self.setClipToView(True)
        self.enableAutoRange(axis='y', enable=True)
        self.enableAutoRange(axis='x', enable=False)

        self.model.frame_changed.connect(self.update_view)
        self.model.labels_changed.connect(self.update_regions)

        self.update_view(0)

    # --------------------------------------------------------

    def update_view(self, frame_idx):

        current_time = frame_idx / self.model.fps
        self.frame_line.setPos(current_time)

        # Center the current time
        half_window = self.window_seconds / 2
        t_min = max(0, current_time - half_window)
        t_max = t_min + self.window_seconds

        # If we are at the end of the video, don't go past the max
        max_time = len(self.time) / self.model.fps
        if t_max > max_time:
            t_max = max_time
            t_min = max(0, t_max - self.window_seconds)

        self.setXRange(t_min, t_max, padding=0)

    # --------------------------------------------------------

    def update_regions(self):
        """
        Draw shaded regions for labels.
        """

        # Remove old regions
        for r in self.region_items:
            self.removeItem(r)
        self.region_items.clear()

        for _, row in self.model.labels.iterrows():

            start_time = row["start"] / self.model.fps
            end_time = row["end"] / self.model.fps

            region = pg.LinearRegionItem(
                values=(start_time, end_time),
                brush=(255, 0, 0, 60),
                movable=False
            )

            # Keep region behind curves and cursor
            region.setZValue(-10)

            self.addItem(region)
            self.region_items.append(region)


class LabelTable(QtWidgets.QTableWidget):

    def __init__(self, model):
        super().__init__()
        self.model = model

        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Start", "End", "Category"])

        self.model.labels_changed.connect(self.refresh)

        self.cellDoubleClicked.connect(self.jump_to_label)

    def refresh(self):
        df = self.model.labels
        self.setRowCount(len(df))

        for i, row in df.iterrows():
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(str(row["start"])))
            self.setItem(i, 1, QtWidgets.QTableWidgetItem(str(row["end"])))
            self.setItem(i, 2, QtWidgets.QTableWidgetItem(str(row["category"])))

    def jump_to_label(self, row, col):
        start = int(self.item(row, 0).text())
        self.model.set_frame(start)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, video_path, tracking_csv):
        super().__init__()

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.model = SessionModel(video_path, tracking_csv)

        self.video = VideoWidget(self.model)
        self._is_playing = False
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)
        interval_ms = int(1000 / self.model.fps) if self.model.fps > 0 else 33
        self.play_timer.setInterval(interval_ms)
        self.video.resize(512, 512)
        
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.model.total_frames - 1)
        self.frame_slider.setValue(self.model.current_frame)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(10)  
        self.frame_slider.setTracking(True)
        self.frame_slider.valueChanged.connect(self.model.set_frame)
        self.model.frame_changed.connect(self._update_slider)
        
        self.slider_label_frame = QtWidgets.QLabel(f"{self.model.current_frame}/{self.model.total_frames}")
        self.model.frame_changed.connect(lambda idx: self.slider_label_frame.setText(f"{idx}/{self.model.total_frames}"))

        self.slider_label_time = QtWidgets.QLabel(f"{self.frame_to_time_string(0)}/{self.frame_to_time_string(self.model.total_frames)}")
        self.model.frame_changed.connect(lambda idx: self.slider_label_time.setText(f"{self.frame_to_time_string(idx)}/{self.frame_to_time_string(self.model.total_frames)}"))

        self.plot = TimeSeriesWidget(self.model)
        self.table = LabelTable(self.model)

        add_label_btn = QtWidgets.QPushButton("Add Label")
        add_label_btn.clicked.connect(self.add_label_dialog)

        save_btn = QtWidgets.QPushButton("Save Labels")
        save_btn.clicked.connect(self.save_labels)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.video)
        top.addWidget(self.plot)

        slider = QtWidgets.QHBoxLayout()
        slider.addWidget(self.slider_label_time)
        slider.addWidget(self.frame_slider)
        slider.addWidget(self.slider_label_frame)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(slider)
        layout.addWidget(self.table)
        layout.addWidget(add_label_btn)
        layout.addWidget(save_btn)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.resize(1200, 900)

    def frame_to_time_string(self, frame_idx):
        total_seconds = frame_idx / self.model.fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def _update_slider(self, frame_idx):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)
        
    def _advance_frame(self):

        next_frame = self.model.current_frame + 1

        if next_frame >= self.model.total_frames:
            self.pause()
            return

        self.model.set_frame(next_frame)

    def play(self):
        if self._is_playing:
            return
        self._is_playing = True
        self.play_timer.start()


    def pause(self):
        if not self._is_playing:
            return
        self._is_playing = False
        self.play_timer.stop()

    def toggle_play(self):
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def add_label_dialog(self):

        start, ok1 = QtWidgets.QInputDialog.getInt(
            self, "Start Frame", "Start:",
            value=self.model.current_frame,
            min=0,
            max=self.model.total_frames
        )
        if not ok1:
            return

        end, ok2 = QtWidgets.QInputDialog.getInt(
            self, "End Frame", "End:",
            value=start+10,
            min=start,
            max=self.model.total_frames
        )
        if not ok2:
            return

        category, ok3 = QtWidgets.QInputDialog.getText(
            self, "Category", "Category:"
        )
        if not ok3:
            return

        self.model.add_label(start, end, category)

    def save_labels(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Labels", "", "CSV (*.csv)"
        )
        if path:
            self.model.save_labels(path)

    def keyPressEvent(self, event):

        if event.isAutoRepeat():
            return

        key = event.key()
        modifiers = event.modifiers()

        step = 1

        if modifiers & QtCore.Qt.ControlModifier:
            step = 10

        if key == QtCore.Qt.Key_Right:
            self.pause()
            self.model.set_frame(self.model.current_frame + step)

        elif key == QtCore.Qt.Key_Left:
            self.pause()
            self.model.set_frame(self.model.current_frame - step)

        elif key == QtCore.Qt.Key_H:
            self.video.toggle_overlay_visibility()

        elif key == QtCore.Qt.Key_Space:
            self.toggle_play()
        
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)

    video_path = "/home/martin/Desktop/DATA/WT/danieau/eyes/00_07dpf_WT_Thu_11_Dec_2025_11h23min15sec_fish_0_eyes.mp4"
    tracking_csv = "/home/martin/Downloads/11-44-00/video_preds/00_07dpf_WT_Thu_11_Dec_2025_11h23min15sec_fish_0_eyes.csv"

    win = MainWindow(video_path, tracking_csv)
    win.show()

    sys.exit(app.exec_())