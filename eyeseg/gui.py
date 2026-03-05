import sys
from enum import Enum

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPen, QBrush, QColor, QImage, QPixmap, QPainter, QKeySequence
from PyQt5.QtWidgets import QStyledItemDelegate, QComboBox
import pyqtgraph as pg

from scipy.signal import savgol_filter
import cv2
import pandas as pd
import numpy as np

class LabelCategory(Enum):
    EYE_CONVERGENCE = "eye convergence"
    PURSUIT_CW = "pursuit cw"
    PURSUIT_CCW = "pursuit ccw"
    SPONTANEOUS_SQUARE = "spontaneous square"

LABEL_COLOR = {
    LabelCategory.EYE_CONVERGENCE: (255, 200, 120, 120),
    LabelCategory.PURSUIT_CW: (144, 238, 144, 120),
    LabelCategory.PURSUIT_CCW: (144, 144, 238, 120),
    LabelCategory.SPONTANEOUS_SQUARE: (186, 85, 211, 120)
}

DIVERGING_4 = (
    (178, 24, 43),    # strong red
    (239, 138, 98),   # light red
    (33, 102, 172),   # strong blue
    (103, 169, 207),  # light blue
)

class InteractionState(Enum):
    IDLE = 0
    ADDING_LABEL = 1

def get_eye_angles_from_keypoints(
        tracking: pd.DataFrame,
        likelihood_threshold: float = 0.9
    ):

    def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray):
        cos_angle = np.sum(v1 * v2, axis=1)
        sin_angle = np.cross(v1, v2)
        angle = np.arctan2(sin_angle, cos_angle)
        return angle

    left_front = tracking.heatmap_tracker.eye_left_front[['x', 'y']].to_numpy()
    left_back = tracking.heatmap_tracker.eye_left_back[['x', 'y']].to_numpy()
    right_front = tracking.heatmap_tracker.eye_right_front[['x', 'y']].to_numpy()
    right_back = tracking.heatmap_tracker.eye_right_back[['x', 'y']].to_numpy()

    left_front_likelihood = tracking.heatmap_tracker.eye_left_front.likelihood.to_numpy()
    left_back_likelihood = tracking.heatmap_tracker.eye_left_back.likelihood.to_numpy()
    right_front_likelihood = tracking.heatmap_tracker.eye_right_front.likelihood.to_numpy()
    right_back_likelihood = tracking.heatmap_tracker.eye_right_back.likelihood.to_numpy()

    # origin top-left
    left_vector = left_back - left_front  
    right_vector = right_back - right_front

    left = compute_angle_between_vectors(left_vector, np.array([0,1]))
    right = compute_angle_between_vectors(right_vector, np.array([0,1]))

    left[(left_front_likelihood < likelihood_threshold) | (left_back_likelihood < likelihood_threshold)] = np.nan
    right[(right_front_likelihood < likelihood_threshold) | (right_back_likelihood < likelihood_threshold)] = np.nan

    return np.rad2deg(left), np.rad2deg(right)

class SessionModel(QtCore.QObject):

    frame_changed = QtCore.pyqtSignal(int)
    labels_changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.cap: cv2.VideoCapture | None = None
        self.video_path: str | None = None
        self.total_frames: int = 0
        self.fps: float = 30.0

        self._current_frame = 0
        self._last_read_frame = -1
        self.saved = True

        self.tracking: pd.DataFrame | None = None
        self.left: np.ndarray | None = None
        self.right: np.ndarray | None = None
        self.time: np.ndarray | None = None
        self.left_smooth: np.ndarray | None = None
        self.right_smooth: np.ndarray | None = None

        self.labels = pd.DataFrame(columns=["start", "end", "category"])

    def load_video(self, path: str):

        if self.cap:
            self.cap.release()

        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._current_frame = 0
        self._last_read_frame = -1
        self.frame_changed.emit(0)

    def load_tracking(self, path: str):
        self.tracking = pd.read_csv(path, header=[0,1,2])
        self.frame_changed.emit(self._current_frame)
        self.left, self.right = get_eye_angles_from_keypoints(self.tracking)
        self.left_smooth = savgol_filter(self.left, window_length=21, polyorder=2)
        self.right_smooth = savgol_filter(self.right, window_length=21, polyorder=2)
        n = len(self.left)
        self.time = np.arange(n) / self.fps

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

        if self.cap is None:
            return

        if self._current_frame == self._last_read_frame + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
            ret, frame = self.cap.read()

        if not ret:
            return None

        self._last_read_frame = self._current_frame
        return frame

    def add_label(self, start: int, end: int, category: LabelCategory):
        new_row = {"start": start, "end": end, "category": category}
        self.labels = pd.concat(
            [self.labels, pd.DataFrame([new_row])],
            ignore_index=True,
        )

        self.saved = False
        self.labels_changed.emit()

    def edit_label(
            self, 
            index: int, 
            start: int|None = None, 
            end: int|None = None, 
            category: LabelCategory|None = None
        ):

        if index < 0 or index >= len(self.labels):
            return

        # Work on a copy
        row = self.labels.iloc[index].copy()

        if start is not None:
            row["start"] = int(start)
        if end is not None:
            row["end"] = int(end)
        if category is not None:
            row["category"] = category 

        self.labels.iloc[index] = row
        
        self.saved = False
        self.labels_changed.emit()

    def delete_label(self, index):
        self.labels = self.labels.drop(index).reset_index(drop=True)
        self.saved = False
        self.labels_changed.emit()

    def save_labels(self, path):
        self.labels.to_csv(path, index=False)
        self.saved = True

class VideoWidget(QtWidgets.QGraphicsView):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.keypoint_items = []
        self.overlay_visible = True

        self.model.frame_changed.connect(self.update_frame)

        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)

        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setFocusPolicy(QtCore.Qt.NoFocus)
        
        self.scene.setBackgroundBrush(QtCore.Qt.transparent)
        self.setStyleSheet("background: transparent")
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.setFixedSize(512, 512)

        self.update_frame(0)

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
            colors = DIVERGING_4,
            likelihood_threshold = 0.9       
        ):

        # Clear old keypoints
        for item in self.keypoint_items:
            self.scene.removeItem(item)
        self.keypoint_items.clear()

        if not self.overlay_visible:
            return
        
        if self.model.tracking is None:
            return

        row = self.model.tracking.heatmap_tracker.iloc[frame_idx]
        count = 0
        for side in ['left', 'right']:
            points = []
            for position in ['front', 'back']:
                x = row[f"eye_{side}_{position}"].x
                y = row[f"eye_{side}_{position}"].y
                l = row[f"eye_{side}_{position}"].likelihood
                points.append((x,y,l))
                
                if l > likelihood_threshold:
                    ellipse = self.scene.addEllipse(
                        x - r, y - r, 2*r, 2*r,
                        pen=QPen(QColor(*colors[count])),
                        brush=QBrush(QColor(*colors[count]))
                    )
                    ellipse.setZValue(2)
                    self.keypoint_items.append(ellipse)
                count += 1

            if (points[0][2] > likelihood_threshold) & (points[1][2] > likelihood_threshold):  
                line = self.scene.addLine(
                    points[0][0],points[0][1],points[1][0],points[1][1],
                    pen=QPen(QColor(*colors[count-1]))
                )
                line.setZValue(1)
                self.keypoint_items.append(line)

    def toggle_overlay_visibility(self):
        self.overlay_visible = not self.overlay_visible
        self.update_overlay(self.model.current_frame)
        
class TimeSeriesWidget(pg.PlotWidget):

    def __init__(
            self, 
            model, 
            window_seconds=15.0,
            colors = DIVERGING_4
        ):
        super().__init__()
    
        self.model = model
        self.window_seconds = float(window_seconds)
        self.region_items = []
        self.show_smooth = True  

        self.left_curve = self.plot([0], [0], pen=pg.mkPen(*colors[1]))
        self.right_curve = self.plot([0], [0], pen=pg.mkPen(*colors[3]))
        self.frame_line = pg.InfiniteLine(
            angle=90, 
            movable=False,
            pen=pg.mkPen(color=(177, 177, 177), width=1, style=QtCore.Qt.DotLine)
        )
        self.zero_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(color=(177, 177, 177), width=1, style=QtCore.Qt.DotLine)
        )  
        self.frame_line.setZValue(-1)      
        self.zero_line.setZValue(-1)
        self.addItem(self.frame_line)
        self.addItem(self.zero_line)

        self.setDownsampling(auto=True)
        self.setClipToView(True)
        self.enableAutoRange(axis='y', enable=False)
        self.enableAutoRange(axis='x', enable=False)
        self.setYRange(-70, 70, padding=0)
        self.setMouseEnabled(x=False, y=False)

        self.model.frame_changed.connect(self.update_view)
        self.model.labels_changed.connect(self.update_regions)

        self.update_view(0)

    def update_curve_data(self):
        left_data = self.model.left_smooth if self.show_smooth else self.model.left
        right_data = self.model.right_smooth if self.show_smooth else self.model.right
        self.left_curve.setData(self.model.time, left_data)
        self.right_curve.setData(self.model.time, right_data)
        
    def update_view(self, frame_idx):

        if self.model.time is None:
            return

        current_time = frame_idx / self.model.fps
        self.frame_line.setPos(current_time)

        # Center the current time
        half_window = self.window_seconds / 2
        t_min = max(0, current_time - half_window)
        t_max = t_min + self.window_seconds

        # If we are at the end of the video, don't go past the max
        max_time = len(self.model.time) / self.model.fps
        if t_max > max_time:
            t_max = max_time
            t_min = max(0, t_max - self.window_seconds)

        self.setXRange(t_min, t_max, padding=0)

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

            region = QtWidgets.QGraphicsRectItem(start_time, -70, end_time - start_time, 140)
            region.setBrush(QColor(*LABEL_COLOR[row.category]))
            region.setPen(QColor(0, 0, 0, 0))      
            region.setZValue(-10) 

            self.addItem(region)
            self.region_items.append(region)

class CategoryDelegate(QStyledItemDelegate):

    def __init__(self, categories, parent=None):
        super().__init__(parent)
        self.categories = [c.value for c in categories]

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.categories)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, QtCore.Qt.EditRole)
        if value in self.categories:
            editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), QtCore.Qt.EditRole)

class LabelTable(QtWidgets.QTableWidget):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._updating = False

        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Start", "End", "Category"])
        self.setItemDelegateForColumn(2, CategoryDelegate(LabelCategory))

        self.model.labels_changed.connect(self.refresh)
        self.cellDoubleClicked.connect(self.jump_to_label)

        self.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked |
                             QtWidgets.QAbstractItemView.SelectedClicked)
        self.itemChanged.connect(self._on_item_changed)

        self.delete_shortcut = QtWidgets.QShortcut(QKeySequence("Delete"), self)
        self.delete_shortcut.activated.connect(self.delete_selected_rows)

    def refresh(self):
        self._updating = True
        df = self.model.labels
        self.setRowCount(len(df))

        for i, row in df.iterrows():
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(str(row["start"])))
            self.setItem(i, 1, QtWidgets.QTableWidgetItem(str(row["end"])))
            self.setItem(i, 2, QtWidgets.QTableWidgetItem(str(row["category"])))

        self.resizeColumnsToContents()
        self._updating = False

    def jump_to_label(self, row, col):
        start = int(self.item(row, 0).text())
        self.model.set_frame(start)

    def _on_item_changed(self, item):

        if self._updating:
            return

        row = item.row()
        col = item.column()

        start = end = category = None

        if col == 0:
            try:
                start = int(item.text())
            except ValueError:
                return
        elif col == 1:
            try:
                end = int(item.text())
            except ValueError:
                return
        elif col == 2:
            category = LabelCategory(item.text())

        self.model.edit_label(row, start=start, end=end, category=category)

    def delete_selected_rows(self):
        selected = set(idx.row() for idx in self.selectedIndexes())
        for row in sorted(selected, reverse=True):  
            self.model.delete_label(row)

class StateInfoWidget(QtWidgets.QFrame):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

        layout = QtWidgets.QVBoxLayout(self)

        self.state_label = QtWidgets.QLabel()
        self.step_label = QtWidgets.QLabel()
        self.commands_label = QtWidgets.QLabel()

        self.state_label.setStyleSheet("font-weight: bold;")
        self.commands_label.setStyleSheet("color: gray;")

        layout.addWidget(self.state_label)
        layout.addWidget(self.commands_label)
        layout.addWidget(self.step_label)
        layout.addStretch()

        self.set_state(InteractionState.IDLE)
        self.set_step(10)

        self.setFocusPolicy(QtCore.Qt.NoFocus)

    def set_step(self, step: int):
        self.step_label.setText(f"STEP: {step}")

    def set_state(self, state: InteractionState):

        if state == InteractionState.IDLE:
            self.state_label.setText("STATE: IDLE")

            self.commands_label.setText(
                "L → Add label \n"
                "← → Move frame \n"
                "Ctrl+← → Move STEP frames \n"
                "Space → Play/Pause \n"
                "H → Hide overlay \n"
                "S → Set step size \n"
                "M → Toggle smoothing"
            )

        elif state == InteractionState.ADDING_LABEL:
            self.state_label.setText("STATE: ADDING LABEL")

            self.commands_label.setText(
                "ENTER → Confirm label \n"
                "ESC → Cancel \n"
                "← → Adjust range \n"
                "Ctrl+← → Adjust range by STEP frames \n"
                "Space → Play (region grows) \n"
                "H → Hide overlay \n"
                "S → Set step size \n"
                "M → Toggle smoothing"
            )

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self._label_region = None
        self._label_category = None
        self._state = InteractionState.IDLE
        self._step = 10

        self.model = SessionModel()
        self.model.frame_changed.connect(self._update_label_region)
        self.state_panel = StateInfoWidget()
        
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
        self.frame_slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.model.frame_changed.connect(self._update_slider)
        
        self.slider_label_frame = QtWidgets.QLabel(f"{self.model.current_frame}/{self.model.total_frames}")
        self.model.frame_changed.connect(lambda idx: self.slider_label_frame.setText(f"{idx}/{self.model.total_frames}"))

        self.slider_label_time = QtWidgets.QLabel(f"{self.frame_to_time_string(0)}/{self.frame_to_time_string(self.model.total_frames)}")
        self.model.frame_changed.connect(lambda idx: self.slider_label_time.setText(f"{self.frame_to_time_string(idx)}/{self.frame_to_time_string(self.model.total_frames)}"))

        self.plot = TimeSeriesWidget(self.model)
        self.table = LabelTable(self.model)

        self._create_menu()

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.video)
        top.addWidget(self.plot)

        slider = QtWidgets.QHBoxLayout()
        slider.addWidget(self.slider_label_time)
        slider.addWidget(self.frame_slider)
        slider.addWidget(self.slider_label_frame)

        bottom =  QtWidgets.QHBoxLayout()
        bottom.addWidget(self.table)
        bottom.addWidget(self.state_panel)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(slider)
        layout.addLayout(bottom)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.resize(1280, 720)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()

    def _create_menu(self):

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Load Video
        load_video_action = QtWidgets.QAction("Load Video...", self)
        load_video_action.triggered.connect(self.load_video)
        file_menu.addAction(load_video_action)

        # Load Tracking
        load_tracking_action = QtWidgets.QAction("Load Tracking CSV...", self)
        load_tracking_action.triggered.connect(self.load_tracking)
        file_menu.addAction(load_tracking_action)

        file_menu.addSeparator()

        # Save Labels
        save_labels_action = QtWidgets.QAction("Save Labels...", self)
        save_labels_action.triggered.connect(self.save_labels)
        save_labels_action.setShortcut(QKeySequence("Ctrl+S"))
        save_labels_action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        file_menu.addAction(save_labels_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("Help")

        shortcuts_action = QtWidgets.QAction("Keyboard Shortcuts...", self)
        shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)

    def show_shortcuts_dialog(self):

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setModal(True)
        dialog.resize(400, 300)

        layout = QtWidgets.QVBoxLayout(dialog)

        table = QtWidgets.QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        shortcuts = [
            ("Right Arrow", "Next frame"),
            ("Left Arrow", "Previous frame"),
            ("Ctrl + Right", f"Forward {self._step} frames"),
            ("Ctrl + Left", f"Backward {self._step} frames"),
            ("Ctrl + s", "Save labels CSV"),
            ("Ctrl + q", "Exit"),
            ("Space", "Play / Pause"),
            ("H", "Toggle overlay visibility"),
            ("S", "Set step size"),
            ("L", "Add label"),
            ("M", "Toggle smoothing"),
            ("Delete", "Delete selected label(s)")
        ]

        table.setRowCount(len(shortcuts))

        for row, (key, action) in enumerate(shortcuts):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(action))

        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

        layout.addWidget(table)
        dialog.exec_()

    def set_state(self, new_state: InteractionState):

        if self._state == new_state:
            return

        self._state = new_state
        self.state_panel.set_state(new_state)

    def closeEvent(self, event):

        if self.model.saved:
            event.accept()
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Save Labels?",
            "You have unsaved labels.\n\nDo you want to save them before exiting?",
            QtWidgets.QMessageBox.Save |
            QtWidgets.QMessageBox.Discard |
            QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save
        )

        if reply == QtWidgets.QMessageBox.Save:
            if not self.save_labels():
                event.ignore()
                return

            event.accept()

        elif reply == QtWidgets.QMessageBox.Discard:
            event.accept()

        else:  # Cancel
            event.ignore()

    def load_video(self):

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Video",
            "",
            "Video Files (*.mp4 *.avi *.mov)"
        )

        if not path:
            return

        self.pause()
        self.model.load_video(path)
        self.video.update_frame(self.model.current_frame)
        self.plot.update_view(self.model.current_frame)
        self.frame_slider.setMaximum(self.model.total_frames - 1)

    def load_tracking(self):

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Tracking CSV",
            "",
            "CSV Files (*.csv)"
        )

        if not path:
            return

        self.model.load_tracking(path)   
        self.plot.update_curve_data()
        self.plot.update_view(self.model.current_frame)
        self.video.update_frame(self.model.current_frame)

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

        if self._state != InteractionState.IDLE:
            return

        self.set_state(InteractionState.ADDING_LABEL)
        
        category, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Category",
            "Category:",
            [c.value for c in LabelCategory],
            current=0,
            editable=False
        )

        if not ok:
            self.set_state(InteractionState.IDLE)
            return

        self._label_category = LabelCategory(category)

        # Initial region around current frame
        current_time = self.model.current_frame / self.model.fps
        duration = 1.0  # default 1 second

        region = pg.LinearRegionItem(
            values=(current_time, current_time + duration),
            movable=True,
            brush=(200, 200, 200, 50)
        )

        region.setZValue(100)
        self.plot.addItem(region)

        self._label_region = region

    def save_labels(self) -> bool:
        
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Labels", "", "CSV (*.csv)"
        )
        
        if path:
            self.model.save_labels(path)
            return True
        
        return False

    def _confirm_label(self):

        if self._state != InteractionState.ADDING_LABEL:
            return

        region = self._label_region
        t_min, t_max = region.getRegion()

        start_frame = int(t_min * self.model.fps)
        end_frame = int(t_max * self.model.fps)

        start_frame = max(0, min(start_frame, self.model.total_frames - 1))
        end_frame = max(start_frame, min(end_frame, self.model.total_frames - 1))

        self.model.add_label(start_frame, end_frame, self._label_category)

        self.plot.removeItem(region)
        self._label_region = None
        self._label_category = None
        self.set_state(InteractionState.IDLE)

    def _cancel_label(self):
        
        if self._state != InteractionState.ADDING_LABEL:
            return

        if self._label_region:
            self.plot.removeItem(self._label_region)

        self._label_region = None
        self._label_category = None

        self.set_state(InteractionState.IDLE)

    def _update_label_region(self, frame_idx):

        if self._state != InteractionState.ADDING_LABEL:
            return

        if self._label_region is None:
            return

        current_time = frame_idx / self.model.fps
        t_min, t_max = self._label_region.getRegion()
        self._label_region.setRegion((t_min, current_time))
            
    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        step = 1

        if key == QtCore.Qt.Key_Return and self._label_region is not None:
            if self._state == InteractionState.ADDING_LABEL:
                self._confirm_label()
                return

        elif key == QtCore.Qt.Key_Escape and self._label_region is not None:
            if self._state == InteractionState.ADDING_LABEL:
                self._cancel_label()
                return

        elif key == QtCore.Qt.Key_S:
            step, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Set Step Size",
                "Enter frame step size:",
                value=self._step,
                min=1,
                max=100,
                step=10
            )
            if ok:
                self._step = step
                self.state_panel.set_step(self._step)
                
        if modifiers & QtCore.Qt.ControlModifier:
            step = self._step

        if key == QtCore.Qt.Key_Right:
            self.pause()
            self.model.set_frame(self.model.current_frame + step)

        elif event.key() == QtCore.Qt.Key_M:
            self.plot.show_smooth = not self.plot.show_smooth
            self.plot.update_curve_data()
            
        elif key == QtCore.Qt.Key_Left:
            self.pause()
            self.model.set_frame(self.model.current_frame - step)

        elif key == QtCore.Qt.Key_H:
            self.video.toggle_overlay_visibility()

        elif key == QtCore.Qt.Key_L:
            if self._state == InteractionState.IDLE:
                self.add_label_dialog()

        elif key == QtCore.Qt.Key_Space:
            self.toggle_play()
        
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())