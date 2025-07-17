import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QSlider, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFileDialog, QComboBox, QLineEdit, QDialog,
    QDialogButtonBox, QFormLayout, QMessageBox, QSizePolicy, QToolButton, QMenu, QStyle,
    QInputDialog, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, QSize, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QColor, QPen, QFont, QCursor, QPolygon, QIntValidator
import json
import os
import random
from datetime import datetime
from collections import deque

# Функция для поиска доступных камер
def find_available_cameras(max_to_check=5):
    available_cameras = []
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Улучшенный класс для обнаружения лазерной точки на основе гауссова профиля
class GaussianLaserDetector:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            available_cams = find_available_cameras()
            if available_cams:
                self.cap = cv2.VideoCapture(available_cams[0])
            else:
                raise RuntimeError("No cameras available")
        
        # Получаем реальные размеры кадра
        self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        
        # Параметры алгоритма
        self.min_brightness = 43
        self.min_red_ratio = 1.8
        self.min_gaussian_score = 0.7
        self.max_point_size = 20
        
        # Фоновая модель
        self.bg_model = None
        self.bg_frames = 30
        self.bg_alpha = 0.95
        self.frame_count = 0
        
        # Стабилизация
        self.tracked_points = deque(maxlen=10)
        self.last_position = None
        self.missed_frames = 0
        self.max_missed_frames = 5
        
        self.is_active = True
        
        # Параметры калибровки
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_target = None
        self.homography_matrix = None
        self.calibration_step = 0
        self.calibration_data = {
            'center': None,
            'top_left': None,
            'top_right': None,
            'bottom_left': None,
            'bottom_right': None
        }
        
        # Коррекция позиции
        self.position_correction = (0, 0)
        self.calibration_complete = False
        
        # Фильтр Калмана
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kalman.statePre = np.zeros((4, 1), np.float32)
        self.kalman.statePost = np.zeros((4, 1), np.float32)

    def update_background(self, frame):
        """Обновление фоновой модели"""
        if self.bg_model is None:
            self.bg_model = frame.copy().astype(np.float32)
            return
            
        # Экспоненциальное скользящее среднее
        cv2.accumulateWeighted(frame, self.bg_model, self.bg_alpha)

    def analyze_gaussian_profile(self, roi):
        """Анализ распределения яркости на соответствие гауссову профилю"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Проверка, что область достаточно большая
        if width < 3 or height < 3:
            return 0.0
        
        # Вертикальный профиль
        vertical_profile = np.mean(gray, axis=1)
        # Горизонтальный профиль
        horizontal_profile = np.mean(gray, axis=0)
        
        def gaussian_fit(profile):
            """Оценка соответствия гауссову распределению"""
            n = len(profile)
            if n < 3:
                return 0.0
                
            # Находим пик
            peak_idx = np.argmax(profile)
            peak_val = profile[peak_idx]
            
            # Рассчитываем стандартное отклонение
            mean = np.sum(np.arange(n) * profile) / np.sum(profile)
            variance = np.sum(profile * (np.arange(n) - mean)**2) / np.sum(profile)
            std = np.sqrt(variance)
            
            # Идеальное гауссово распределение
            x = np.arange(n)
            ideal = np.exp(-(x - mean)**2 / (2 * std**2)) * peak_val
            
            # Сравнение с реальным профилем
            correlation = np.corrcoef(profile, ideal)[0, 1]
            return max(0, correlation)  # Корреляция от 0 до 1
        
        vertical_score = gaussian_fit(vertical_profile)
        horizontal_score = gaussian_fit(horizontal_profile)
        
        return (vertical_score + horizontal_score) / 2

    def detect(self, frame):
        """Основная функция обнаружения лазерной точки"""
        # Работа в HSV пространстве
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Маска по яркости
        _, bright_mask = cv2.threshold(v, self.min_brightness, 255, cv2.THRESH_BINARY)
        
        # Морфологическая обработка
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Кандидаты на лазерную точку
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1 or area > self.max_point_size**2:
                continue
                
            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)
            if w < 2 or h < 2:
                continue
                
            # Проверяем цветовое соотношение
            roi = frame[y:y+h, x:x+w]
            mean_bgr = cv2.mean(roi)[:3]
            b, g, r = mean_bgr
            
            # Красный должен доминировать
            if r < max(b, g) * self.min_red_ratio:
                continue
                
            # Анализ гауссова профиля
            gauss_score = self.analyze_gaussian_profile(roi)
            if gauss_score < self.min_gaussian_score:
                continue
                
            # Центр масс
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                candidates.append((cx, cy, gauss_score, r))
        
        # Выбираем лучшего кандидата
        if candidates:
            # Сортируем по оценке гауссова профиля и яркости
            candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
            return candidates[0][:2]  # возвращаем только координаты
        
        return None

    def process_frame(self):
        if not self.is_active:
            return None, None, None
            
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
            
        # Обновление фона
        self.update_background(frame)
        self.frame_count += 1
        
        # Пропускаем первые кадры для инициализации фона
        if self.frame_count < self.bg_frames:
            return frame, None, None
            
        best_spot = self.detect(frame)
        display_frame = frame.copy()
        
        if best_spot:
            cx, cy = best_spot
            
            # Применяем коррекцию позиции
            cx += self.position_correction[0]
            cy += self.position_correction[1]
            
            # Применяем фильтр Калмана для сглаживания
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            self.kalman.correct(measurement)
            prediction = self.kalman.predict()
            cx, cy = int(prediction[0]), int(prediction[1])
            
            if self.calibration_mode:
                # В режиме калибровки не используем стабилизацию
                cv2.circle(display_frame, (cx, cy), 10, (0, 255, 0), 2)
                laser_pos = (cx, cy)
                self.calibration_points.append((cx, cy))
            else:
                self.tracked_points.append((cx, cy))
                avg_x = int(np.mean([p[0] for p in self.tracked_points]))
                avg_y = int(np.mean([p[1] for p in self.tracked_points]))
                self.last_position = (avg_x, avg_y)
                cv2.circle(display_frame, (avg_x, avg_y), 10, (0, 255, 0), 2)
                laser_pos = (avg_x, avg_y)
                self.missed_frames = 0
        elif self.last_position and self.missed_frames < self.max_missed_frames:
            laser_pos = self.last_position
            self.missed_frames += 1
        else:
            laser_pos = None
            self.missed_frames += 1
            
        return display_frame, laser_pos, None
    
    def start_calibration(self, target):
        self.calibration_mode = True
        self.calibration_target = target
        self.calibration_points = []
        self.calibration_step += 1
    
    def finish_calibration(self):
        if not self.calibration_points:
            return False
            
        avg_x = np.mean([p[0] for p in self.calibration_points])
        avg_y = np.mean([p[1] for p in self.calibration_points])
        
        self.calibration_data[self.calibration_target] = (avg_x, avg_y)
        self.calibration_mode = False
        
        # Если все точки собраны, вычисляем матрицу гомографии
        if all(self.calibration_data.values()):
            self.compute_homography()
            self.calibration_complete = True
        
        return True
    
    def compute_homography(self):
        # Исходные точки (с камеры)
        src_points = np.array([
            self.calibration_data['top_left'],
            self.calibration_data['top_right'],
            self.calibration_data['bottom_right'],
            self.calibration_data['bottom_left']
        ], dtype=np.float32)
        
        # Целевые точки (экранные координаты)
        dst_points = np.array([
            [0, 0],                 # top_left
            [1, 0],                 # top_right
            [1, 1],                 # bottom_right
            [0, 1]                  # bottom_left
        ], dtype=np.float32)
        
        # Вычисляем матрицу гомографии
        self.homography_matrix, _ = cv2.findHomography(src_points, dst_points)
    
    def transform_point(self, point):
        if not self.homography_matrix or point is None:
            return point
        
        # Преобразуем точку с помощью матрицы гомографии
        src_pt = np.array([point[0], [point[1]]], dtype=np.float32)
        dst_pt = cv2.perspectiveTransform(src_pt.reshape(1, -1, 2), self.homography_matrix)
        return (dst_pt[0][0][0], dst_pt[0][0][1])
    
    def set_position_correction(self, dx, dy):
        """Установка коррекции позиции лазера"""
        self.position_correction = (dx, dy)
    
    def save_calibration(self):
        return {
            'calibration_data': self.calibration_data,
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            'position_correction': self.position_correction
        }
    
    def load_calibration(self, data):
        self.calibration_data = data.get('calibration_data', {
            'center': None,
            'top_left': None,
            'top_right': None,
            'bottom_left': None,
            'bottom_right': None
        })
        
        homography_data = data.get('homography_matrix')
        if homography_data:
            self.homography_matrix = np.array(homography_data, dtype=np.float32)
        else:
            self.homography_matrix = None
            
        correction_data = data.get('position_correction', (0, 0))
        self.position_correction = correction_data
        
    def reset_calibration(self):
        """Сброс всей калибровки"""
        self.calibration_data = {
            'center': None,
            'top_left': None,
            'top_right': None,
            'bottom_left': None,
            'bottom_right': None
        }
        self.homography_matrix = None
        self.position_correction = (0, 0)
        self.calibration_complete = False
        self.calibration_mode = False
        self.calibration_step = 0
        self.calibration_points = []
    
    def release(self):
        self.is_active = False
        if self.cap.isOpened():
            self.cap.release()

# Окно авторизации
class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.setFixedSize(400, 300)
        self.setStyleSheet("""
            background-color: #2c3e50;
            color: #ecf0f1;
            font-size: 16px;
        """)
        
        layout = QVBoxLayout()
        
        title = QLabel("LAZER TASK")
        title.setStyleSheet("""
            font-size: 32px; 
            font-weight: bold; 
            color: #3498db;
            margin-bottom: 30px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Enter username")
        self.username_edit.setStyleSheet("""
            padding: 10px;
            border: 1px solid #34495e;
            border-radius: 4px;
            background-color: #34495e;
            color: #ecf0f1;
        """)
        
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setStyleSheet("""
            padding: 10px;
            border: 1px solid #34495e;
            border-radius: 4px;
            background-color: #34495e;
            color: #ecf0f1;
        """)
        
        form_layout.addRow("Username:", self.username_edit)
        form_layout.addRow("Password:", self.password_edit)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        buttons.accepted.connect(self.authenticate)
        buttons.rejected.connect(self.reject)
        
        layout.addLayout(form_layout)
        layout.addWidget(buttons)
        self.setLayout(layout)
        
        # Загрузка пользователей
        self.users = self.load_users()
        if "admin" not in self.users:
            self.users["admin"] = "admin"
    
    def load_users(self):
        try:
            if os.path.exists("users.json"):
                with open("users.json", "r") as f:
                    return json.load(f)
            return {}
        except:
            return {}
    
    def save_users(self):
        with open("users.json", "w") as f:
            json.dump(self.users, f)
    
    def authenticate(self):
        username = self.username_edit.text()
        password = self.password_edit.text()
        
        if not username or not password:
            QMessageBox.warning(self, "Error", "Please enter both username and password")
            return
            
        if username in self.users and self.users[username] == password:
            self.current_user = username
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Invalid username or password")

# Мишень для отображения
class TargetWidget(QLabel):
    CIRCLE = 0
    RECTANGLE = 1
    DIAMOND = 2
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #333; border: 2px solid #555;")
        self.setMouseTracking(True)
        
        # Основная мишень
        self.target_pos = QPoint(400, 300)
        self.target_size = 100
        self.target_color = QColor(255, 0, 0)
        
        # Фон и изображение мишени
        self.bg_image = None
        self.target_image = None
        self.laser_pos = None
        self.laser_detected = False
        
        # Для управления мишенями
        self.targets = []
        self.presets = []
        self.current_preset = 0
        self.selected_target = -1
        self.drag_start_pos = None
        self.drag_target_index = -1
        self.cursor_changed = False
        
        # Для анимации попаданий
        self.hits = []
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)
        
        # Для калибровки
        self.calibration_mode = False
        self.calibration_point = None
        self.calibration_point_name = ""
        self.correction_mode = False
        self.correction_point = None
        self.correction_offset = (0, 0)
        
        # Создаем пресеты
        self.create_presets()
        self.apply_preset(0)
    
    def create_presets(self):
        # Пресет 1: Классическая мишень
        self.presets.append({
            "name": "Classic Target",
            "bg_color": QColor(50, 50, 50),
            "bg_image": None,
            "targets": [
                {"pos": QPoint(400, 300), "size": 100, "color": QColor(255, 0, 0), "shape": self.CIRCLE}
            ]
        })
        
        # Пресет 2: Космическая тема
        self.presets.append({
            "name": "Space Theme",
            "bg_color": QColor(10, 10, 40),
            "bg_image": None,
            "targets": [
                {"pos": QPoint(300, 250), "size": 80, "color": QColor(0, 200, 255), "shape": self.CIRCLE},
                {"pos": QPoint(500, 350), "size": 60, "color": QColor(200, 0, 255), "shape": self.CIRCLE}
            ]
        })
    
    def set_calibration_point(self, point_name, position):
        """Устанавливает точку для калибровки"""
        self.calibration_point = position
        self.calibration_point_name = point_name
        self.update()
    
    def add_preset(self, name, bg_image=None, bg_color=None):
        """Добавляет новый пресет"""
        if bg_color is None:
            bg_color = QColor(50, 50, 50)
            
        new_preset = {
            "name": name,
            "bg_color": bg_color,
            "bg_image": bg_image,
            "targets": [t.copy() for t in self.targets]
        }
        
        self.presets.append(new_preset)
        return len(self.presets) - 1
    
    def apply_preset(self, index):
        if 0 <= index < len(self.presets):
            self.current_preset = index
            preset = self.presets[index]
            self.targets = [t.copy() for t in preset["targets"]]
            
            if preset["bg_image"]:
                self.bg_image = QPixmap(preset["bg_image"])
            else:
                self.bg_image = None
                
            self.setStyleSheet(f"background-color: {preset['bg_color'].name()}; border: 2px solid #555;")
            self.update()
    
    def set_background(self, image_path):
        if image_path:
            self.presets[self.current_preset]["bg_image"] = image_path
            self.bg_image = QPixmap(image_path)
            self.update()
    
    def add_target(self, shape=CIRCLE, pos=None, size=80, color=None):
        """Добавляет новую мишень"""
        if color is None:
            color = QColor(random.randint(50, 255), 
                          random.randint(50, 255), 
                          random.randint(50, 255))
        
        if pos is None:
            pos = QPoint(self.width() // 2, self.height() // 2)
            
        self.targets.append({
            "pos": pos,
            "size": size,
            "color": color,
            "shape": shape
        })
        
        if self.current_preset >= 0:
            self.presets[self.current_preset]["targets"] = [t.copy() for t in self.targets]
        self.update()
    
    def remove_target(self, index):
        """Удаляет мишень по индексу"""
        if 0 <= index < len(self.targets):
            self.targets.pop(index)
            self.selected_target = -1
            if self.current_preset >= 0:
                self.presets[self.current_preset]["targets"] = [t.copy() for t in self.targets]
            self.update()
            return True
        return False
    
    def set_laser_position(self, pos):
        self.laser_pos = pos
        self.update()
    
    def add_hit(self, pos, score):
        self.hits.append({
            'pos': pos,
            'time': datetime.now(),
            'score': score
        })
    
    def update_animation(self):
        now = datetime.now()
        self.hits = [hit for hit in self.hits if (now - hit['time']).total_seconds() < 3]
        self.update()
    
    def get_target_at_position(self, pos):
        for i in range(len(self.targets)-1, -1, -1):
            target = self.targets[i]
            target_pos = target['pos']
            target_size = target['size']
            
            if target_size <= 0:
                continue
                
            if target['shape'] == self.CIRCLE:
                distance = QPoint(pos - target_pos).manhattanLength()
                if distance < target_size:
                    return i
                    
            elif target['shape'] == self.RECTANGLE:
                rect = QRect(
                    target_pos.x() - target_size, 
                    target_pos.y() - target_size,
                    target_size * 2, 
                    target_size * 2
                )
                if rect.contains(pos):
                    return i
                    
            elif target['shape'] == self.DIAMOND:
                # Ромб - это квадрат, повернутый на 45 градусов
                distance = abs(pos.x() - target_pos.x()) + abs(pos.y() - target_pos.y())
                if distance < target_size:
                    return i
        
        return -1
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            
            # Режим коррекции
            if self.correction_mode and self.laser_pos:
                self.correction_point = pos
                self.correction_offset = (
                    pos.x() - self.laser_pos[0],
                    pos.y() - self.laser_pos[1]
                )
                self.update()
                return
                
            # Проверяем, была ли нажата мишень
            target_index = self.get_target_at_position(pos)
            if target_index >= 0:
                self.drag_target_index = target_index
                if self.drag_target_index >= 0:
                    self.drag_start_pos = pos
                    target = self.targets.pop(self.drag_target_index)
                    self.targets.append(target)
                    self.drag_target_index = len(self.targets) - 1
                    self.selected_target = self.drag_target_index
                    self.update()
            else:
                # Клик в пустом пространстве - снимаем выделение
                self.selected_target = -1
                self.update()
    
    def mouseMoveEvent(self, event):
        pos = event.pos()
        
        if self.correction_mode and self.correction_point:
            self.correction_point = pos
            self.correction_offset = (
                pos.x() - self.laser_pos[0],
                pos.y() - self.laser_pos[1]
            )
            self.update()
            return
            
        if self.drag_target_index >= 0 and self.drag_start_pos is not None:
            target = self.targets[self.drag_target_index]
            delta = pos - self.drag_start_pos
            target['pos'] += delta
            self.drag_start_pos = pos
            self.update()
    
    def mouseReleaseEvent(self, event):
        if self.correction_mode and self.correction_point:
            self.correction_point = None
            return
            
        self.drag_target_index = -1
        self.drag_start_pos = None
        
        if self.current_preset >= 0:
            self.presets[self.current_preset]["targets"] = [t.copy() for t in self.targets]
    
    def wheelEvent(self, event):
        # Отключено для предотвращения случайного закрытия
        pass
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        rect = self.rect()
        if self.bg_image:
            scaled_bg = self.bg_image.scaled(rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            painter.drawPixmap(rect, scaled_bg)
        else:
            painter.fillRect(rect, QBrush(self.palette().window().color()))
        
        # Рисуем углы калибровки
        painter.setPen(QPen(QColor(0, 255, 255), 2))
        
        # Верхний левый
        painter.drawLine(20, 20, 40, 20)
        painter.drawLine(20, 20, 20, 40)
        
        # Верхний правый
        painter.drawLine(self.width() - 20, 20, self.width() - 40, 20)
        painter.drawLine(self.width() - 20, 20, self.width() - 20, 40)
        
        # Нижний правый
        painter.drawLine(self.width() - 20, self.height() - 20, self.width() - 40, self.height() - 20)
        painter.drawLine(self.width() - 20, self.height() - 20, self.width() - 20, self.height() - 40)
        
        # Нижний левый
        painter.drawLine(20, self.height() - 20, 40, self.height() - 20)
        painter.drawLine(20, self.height() - 20, 20, self.height() - 40)
        
        # Центр
        center_x = self.width() // 2
        center_y = self.height() // 2
        painter.drawLine(center_x - 20, center_y, center_x + 20, center_y)
        painter.drawLine(center_x, center_y - 20, center_x, center_y + 20)
        painter.drawEllipse(center_x - 5, center_y - 5, 10, 10)
        
        # Рисуем мишени - теперь только контуры без заливки
        for idx, target in enumerate(self.targets):
            target_size = target['size']
            if target_size <= 0:
                continue
                
            # Устанавливаем цвет контура
            painter.setPen(QPen(target['color'], 3))  # Толщина контура 3 пикселя
            painter.setBrush(Qt.NoBrush)  # Убираем заливку
            
            if target['shape'] == self.CIRCLE:
                # Рисуем только кольца без центра
                for i in range(1, 6):
                    radius = target_size * i / 5
                    painter.drawEllipse(target['pos'], radius, radius)
                
            elif target['shape'] == self.RECTANGLE:
                # Рисуем только контур прямоугольника
                rect = QRect(
                    target['pos'].x() - target_size, 
                    target['pos'].y() - target_size,
                    target_size * 2, 
                    target_size * 2
                )
                painter.drawRect(rect)
                
                # Рисуем внутренние прямоугольники
                painter.setPen(QPen(target['color'], 1))
                for i in range(1, 5):
                    size = target_size * (5 - i) / 5
                    inner_rect = QRect(
                        target['pos'].x() - size, 
                        target['pos'].y() - size,
                        size * 2, 
                        size * 2
                    )
                    painter.drawRect(inner_rect)
                
            elif target['shape'] == self.DIAMOND:
                # Рисуем ромб (квадрат, повернутый на 45 градусов)
                points = [
                    QPoint(target['pos'].x(), target['pos'].y() - target_size),
                    QPoint(target['pos'].x() + target_size, target['pos'].y()),
                    QPoint(target['pos'].x(), target['pos'].y() + target_size),
                    QPoint(target['pos'].x() - target_size, target['pos'].y())
                ]
                painter.drawPolygon(QPolygon(points))
                
                # Рисуем внутренние ромбы
                painter.setPen(QPen(target['color'], 1))
                for i in range(1, 5):
                    size = target_size * (5 - i) / 5
                    inner_points = [
                        QPoint(target['pos'].x(), target['pos'].y() - size),
                        QPoint(target['pos'].x() + size, target['pos'].y()),
                        QPoint(target['pos'].x(), target['pos'].y() + size),
                        QPoint(target['pos'].x() - size, target['pos'].y())
                    ]
                    painter.drawPolygon(QPolygon(inner_points))
            
            # Выделение выбранной мишени
            if idx == self.selected_target:
                painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
                if target['shape'] == self.CIRCLE:
                    painter.drawEllipse(target['pos'], target_size + 10, target_size + 10)
                elif target['shape'] == self.RECTANGLE:
                    rect = QRect(
                        target['pos'].x() - target_size - 10, 
                        target['pos'].y() - target_size - 10,
                        (target_size + 10) * 2, 
                        (target_size + 10) * 2
                    )
                    painter.drawRect(rect)
                elif target['shape'] == self.DIAMOND:
                    points = [
                        QPoint(target['pos'].x(), target['pos'].y() - target_size - 10),
                        QPoint(target['pos'].x() + target_size + 10, target['pos'].y()),
                        QPoint(target['pos'].x(), target['pos'].y() + target_size + 10),
                        QPoint(target['pos'].x() - target_size - 10, target['pos'].y())
                    ]
                    painter.drawPolygon(QPolygon(points))
        
        # Рисуем лазерную точку
        if self.laser_pos and self.laser_detected:
            painter.setBrush(QBrush(QColor(255, 0, 0)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(self.laser_pos[0], self.laser_pos[1], 10, 10)
            
            # Рисуем коррекцию
            if self.correction_mode and self.correction_point:
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                painter.drawLine(
                    self.laser_pos[0], self.laser_pos[1],
                    self.correction_point.x(), self.correction_point.y()
                )
                painter.setBrush(QBrush(QColor(255, 255, 0)))
                painter.drawEllipse(self.correction_point, 5, 5)
        
        # Визуальные подсказки для калибровки
        if self.calibration_point:
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawLine(self.calibration_point.x() - 20, self.calibration_point.y(), 
                             self.calibration_point.x() + 20, self.calibration_point.y())
            painter.drawLine(self.calibration_point.x(), self.calibration_point.y() - 20, 
                             self.calibration_point.x(), self.calibration_point.y() + 20)
            painter.drawEllipse(self.calibration_point, 30, 30)
            
            # Текст с инструкцией
            painter.setFont(QFont('Arial', 16))
            painter.setPen(QPen(QColor(255, 255, 0)))
            painter.drawText(self.calibration_point.x() + 40, self.calibration_point.y() - 40, 
                             f"Point laser at {self.calibration_point_name}")
        
        # Рисуем попадания
        now = datetime.now()
        for hit in self.hits:
            time_diff = (now - hit['time']).total_seconds()
            if time_diff < 3:
                alpha = 1.0 - time_diff / 3
                radius = 10 + 20 * time_diff
                color = QColor(0, 255, 0, int(255 * alpha))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(color, 2))
                painter.drawEllipse(QPoint(hit['pos'][0], hit['pos'][1]), radius, radius)
                
                painter.setPen(QPen(color, 1))
                painter.setFont(QFont('Arial', 16))
                text = f"+{hit['score']}"
                painter.drawText(hit['pos'][0] + 15, hit['pos'][1] - 15, text)

# Главное меню
class MainMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)
        
        title = QLabel("LAZER TASK")
        title.setStyleSheet("""
            font-size: 48px; 
            font-weight: bold; 
            color: #4CAF50;
            margin-bottom: 40px;
        """)
        title.setAlignment(Qt.AlignCenter)
        
        btn_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                font-size: 18px;
                margin: 4px 2px;
                border-radius: 8px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        
        start_btn = QPushButton("START")
        start_btn.setStyleSheet(btn_style)
        start_btn.clicked.connect(self.start_game)
        
        options_btn = QPushButton("OPTIONS")
        options_btn.setStyleSheet(btn_style)
        options_btn.clicked.connect(self.show_options)
        
        records_btn = QPushButton("RECORDS")
        records_btn.setStyleSheet(btn_style)
        records_btn.clicked.connect(self.show_records)
        
        logout_btn = QPushButton("LOGOUT")
        logout_btn.setStyleSheet(btn_style)
        logout_btn.clicked.connect(self.logout)
        
        exit_btn = QPushButton("EXIT")
        exit_btn.setStyleSheet(btn_style)
        exit_btn.clicked.connect(QApplication.instance().quit)
        
        layout.addWidget(title)
        layout.addWidget(start_btn)
        layout.addWidget(options_btn)
        layout.addWidget(records_btn)
        layout.addWidget(logout_btn)
        layout.addWidget(exit_btn)
        
        self.setLayout(layout)
    
    def start_game(self):
        self.parent.stacked_widget.setCurrentIndex(1)
    
    def show_options(self):
        self.parent.stacked_widget.setCurrentIndex(2)
    
    def show_records(self):
        self.parent.stacked_widget.setCurrentIndex(3)
    
    def logout(self):
        self.parent.show_login()

# Игровое окно
class GameWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.laser_detector = None
        self.camera_active = False
        self.game_active = False
        self.score = 0
        self.game_time = 60
        self.hit_cooldown = False
        self.camera_fullscreen = False
        self.show_mask = False  # Флаг для отображения маски вместо кадра
        self.initUI()
    
    def initUI(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        
        self.target_widget = TargetWidget()
        main_layout.addWidget(self.target_widget, 1)
        
        self.side_panel = QFrame()
        self.side_panel.setFrameShape(QFrame.StyledPanel)
        self.side_panel.setStyleSheet("""
            background-color: #2c3e50;
            border-left: 1px solid #34495e;
        """)
        self.side_panel.setFixedWidth(300)
        main_layout.addWidget(self.side_panel, 0)
        
        # Главный макет для боковой панели
        main_side_layout = QVBoxLayout()
        main_side_layout.setContentsMargins(0, 0, 0, 0)
        main_side_layout.setSpacing(0)
        self.side_panel.setLayout(main_side_layout)
        
        # Кнопка сворачивания/разворачивания панели
        self.toggle_btn = QPushButton("◀")
        self.toggle_btn.setFixedHeight(30)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_side_panel)
        main_side_layout.addWidget(self.toggle_btn)
        
        # Область с прокруткой для контента
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setAlignment(Qt.AlignTop)
        scroll_layout.setSpacing(15)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        main_side_layout.addWidget(scroll_area)
        
        # Пресеты
        preset_group = QGroupBox("PRESETS")
        preset_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #3498db;
            }
        """)
        preset_layout = QVBoxLayout()
        
        self.preset_combo = QComboBox()
        preset_layout.addWidget(self.preset_combo)
        
        self.new_preset_btn = QPushButton("New Preset")
        self.new_preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 8px;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        self.new_preset_btn.clicked.connect(self.create_new_preset)
        preset_layout.addWidget(self.new_preset_btn)
        
        self.delete_preset_btn = QPushButton("Delete Preset")
        self.delete_preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px;
                border-radius: 4px;
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)
        self.delete_preset_btn.setEnabled(False)
        preset_layout.addWidget(self.delete_preset_btn)
        
        preset_group.setLayout(preset_layout)
        scroll_layout.addWidget(preset_group)
        
        self.update_presets_combo()
        self.preset_combo.currentIndexChanged.connect(self.change_preset)
        
        # Настройки мишени и фона
        target_group = QGroupBox("TARGET & BACKGROUND")
        target_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #3498db;
            }
        """)
        target_layout = QVBoxLayout()
        
        self.bg_btn = QPushButton("Set Background")
        self.bg_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2c3e50;
            }
        """)
        self.bg_btn.clicked.connect(self.select_background)
        target_layout.addWidget(self.bg_btn)
        
        target_type_group = QGroupBox("ADD TARGETS")
        target_type_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #3498db;
                margin-top: 10px;
            }
        """)
        target_type_layout = QHBoxLayout()
        
        circle_btn = QPushButton()
        circle_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        circle_btn.setToolTip("Add Circle Target")
        circle_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                border-radius: 15px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        circle_btn.clicked.connect(lambda: self.add_new_target(TargetWidget.CIRCLE))
        target_type_layout.addWidget(circle_btn)
        
        rect_btn = QPushButton()
        rect_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_FileIcon))
        rect_btn.setToolTip("Add Rectangle Target")
        rect_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                border-radius: 5px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        rect_btn.clicked.connect(lambda: self.add_new_target(TargetWidget.RECTANGLE))
        target_type_layout.addWidget(rect_btn)
        
        diamond_btn = QPushButton()
        diamond_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_ArrowUp))
        diamond_btn.setToolTip("Add Diamond Target")
        diamond_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                border-radius: 5px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        diamond_btn.clicked.connect(lambda: self.add_new_target(TargetWidget.DIAMOND))
        target_type_layout.addWidget(diamond_btn)
        
        target_type_group.setLayout(target_type_layout)
        target_layout.addWidget(target_type_group)
        
        # Управление выбранной мишенью
        target_control_group = QGroupBox("TARGET CONTROL")
        target_control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #3498db;
                margin-top: 10px;
            }
        """)
        target_control_layout = QHBoxLayout()
        
        self.delete_target_btn = QPushButton("Delete Selected")
        self.delete_target_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px;
                border-radius: 4px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.delete_target_btn.setEnabled(False)
        self.delete_target_btn.clicked.connect(self.delete_selected_target)
        target_control_layout.addWidget(self.delete_target_btn)
        
        self.target_size_slider = QSlider(Qt.Horizontal)
        self.target_size_slider.setRange(20, 300)
        self.target_size_slider.setValue(100)
        self.target_size_slider.valueChanged.connect(self.update_target_size)
        target_control_layout.addWidget(self.target_size_slider)
        
        target_control_group.setLayout(target_control_layout)
        target_layout.addWidget(target_control_group)
        
        target_group.setLayout(target_layout)
        scroll_layout.addWidget(target_group)
        
        # Управление камерой
        camera_group = QGroupBox("CAMERA CONTROL")
        camera_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #3498db;
            }
        """)
        camera_layout = QVBoxLayout()
        
        # Кнопка переключения между камерой и маской
        self.toggle_mask_btn = QPushButton("Show Mask")
        self.toggle_mask_btn.setCheckable(True)
        self.toggle_mask_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2c3e50;
            }
            QPushButton:checked {
                background-color: #9b59b6;
            }
        """)
        self.toggle_mask_btn.clicked.connect(self.toggle_mask_view)
        camera_layout.addWidget(self.toggle_mask_btn)
        
        self.camera_toggle = QPushButton("Enable Camera")
        self.camera_toggle.setCheckable(True)
        self.camera_toggle.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2c3e50;
            }
            QPushButton:checked {
                background-color: #27ae60;
            }
        """)
        self.camera_toggle.clicked.connect(self.toggle_camera)
        camera_layout.addWidget(self.camera_toggle)
        
        def create_slider_with_label(label_text, min_val, max_val, default_val, callback, value_suffix=""):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("color: #ecf0f1;")
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default_val)
            slider.valueChanged.connect(callback)
            
            value_label = QLabel(f"{default_val}{value_suffix}")
            value_label.setStyleSheet("color: white; font-weight: bold;")
            value_label.setFixedWidth(40)
            
            slider.valueChanged.connect(lambda v: value_label.setText(f"{v}{value_suffix}"))
            
            layout = QHBoxLayout()
            layout.addWidget(label)
            layout.addWidget(slider)
            layout.addWidget(value_label)
            
            return slider, value_label, layout
        
        self.brightness_slider, self.brightness_value, brightness_layout = create_slider_with_label(
            "Min Brightness", 0, 255, 43, self.set_brightness
        )
        camera_layout.addLayout(brightness_layout)
        
        self.red_ratio_slider, self.red_ratio_value, red_ratio_layout = create_slider_with_label(
            "Red Ratio %", 100, 300, 180, self.set_red_ratio, "%"
        )
        camera_layout.addLayout(red_ratio_layout)
        
        self.gauss_score_slider, self.gauss_score_value, gauss_score_layout = create_slider_with_label(
            "Gauss Score %", 0, 100, 70, self.set_gauss_score, "%"
        )
        camera_layout.addLayout(gauss_score_layout)
        
        self.exposure_slider, self.exposure_value, exposure_layout = create_slider_with_label(
            "Exposure", -10, 10, -7, self.set_exposure
        )
        camera_layout.addLayout(exposure_layout)
        
        self.hide_camera_check = QCheckBox("Hide Camera Preview")
        self.hide_camera_check.stateChanged.connect(self.toggle_camera_visibility)
        camera_layout.addWidget(self.hide_camera_check)
        
        # Расширенная калибровка
        calibration_group = QGroupBox("ADVANCED CALIBRATION")
        calibration_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #3498db;
            }
        """)
        calibration_layout = QVBoxLayout()
        
        calibration_info = QLabel("Calibration steps:\n1. Center\n2. Top Left\n3. Top Right\n4. Bottom Right\n5. Bottom Left")
        calibration_info.setStyleSheet("color: #ecf0f1; font-size: 12px;")
        calibration_layout.addWidget(calibration_info)
        
        self.calibration_btn = QPushButton("Start Calibration")
        self.calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.calibration_btn.clicked.connect(self.start_calibration)
        calibration_layout.addWidget(self.calibration_btn)
        
        self.next_step_btn = QPushButton("Next Calibration Point")
        self.next_step_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.next_step_btn.setEnabled(False)
        self.next_step_btn.clicked.connect(self.next_calibration_step)
        calibration_layout.addWidget(self.next_step_btn)
        
        # Кнопка сброса калибровки
        self.reset_calibration_btn = QPushButton("Reset Calibration")
        self.reset_calibration_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.reset_calibration_btn.clicked.connect(self.reset_calibration)
        calibration_layout.addWidget(self.reset_calibration_btn)
        
        # Коррекция позиции
        self.correction_btn = QPushButton("Position Correction")
        self.correction_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:checked {
                background-color: #8e44ad;
            }
        """)
        self.correction_btn.setCheckable(True)
        self.correction_btn.clicked.connect(self.toggle_correction_mode)
        calibration_layout.addWidget(self.correction_btn)
        
        # Ручная коррекция
        correction_layout = QHBoxLayout()
        
        self.correction_x = QLineEdit("0")
        self.correction_x.setFixedWidth(50)
        self.correction_x.setValidator(QIntValidator(-100, 100))
        self.correction_x.setStyleSheet("background-color: white; color: black;")
        self.correction_x.textChanged.connect(self.update_correction)
        
        self.correction_y = QLineEdit("0")
        self.correction_y.setFixedWidth(50)
        self.correction_y.setValidator(QIntValidator(-100, 100))
        self.correction_y.setStyleSheet("background-color: white; color: black;")
        self.correction_y.textChanged.connect(self.update_correction)
        
        correction_layout.addWidget(QLabel("X:"))
        correction_layout.addWidget(self.correction_x)
        correction_layout.addWidget(QLabel("Y:"))
        correction_layout.addWidget(self.correction_y)
        
        calibration_layout.addLayout(correction_layout)
        
        self.calibration_status = QLabel("Calibration: Not started")
        self.calibration_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        calibration_layout.addWidget(self.calibration_status)
        
        calibration_group.setLayout(calibration_layout)
        camera_layout.addWidget(calibration_group)
        
        camera_group.setLayout(camera_layout)
        scroll_layout.addWidget(camera_group)
        
        # Информация о игре
        info_group = QGroupBox("GAME INFO")
        info_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #3498db;
            }
        """)
        info_layout = QVBoxLayout()
        
        self.score_label = QLabel("Score: 0")
        self.score_label.setStyleSheet("font-size: 16px; color: #e74c3c;")
        info_layout.addWidget(self.score_label)
        
        self.time_label = QLabel("Time: 60s")
        self.time_label.setStyleSheet("font-size: 16px; color: #3498db;")
        info_layout.addWidget(self.time_label)
        
        self.start_btn = QPushButton("Start Game")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.start_btn.clicked.connect(self.start_game)
        info_layout.addWidget(self.start_btn)
        
        back_btn = QPushButton("Back to Menu")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #95a5a6;
            }
        """)
        back_btn.clicked.connect(self.back_to_menu)
        info_layout.addWidget(back_btn)
        
        info_group.setLayout(info_layout)
        scroll_layout.addWidget(info_group)
        
        # Виджет камеры
        self.camera_label = QLabel(self.target_widget)
        self.camera_label.setFixedSize(240, 180)
        self.camera_label.move(20, 20)
        self.camera_label.setStyleSheet("""
            background-color: black; 
            border: 2px solid #3498db;
            border-radius: 4px;
        """)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.hide()
        
        # Кнопка разворачивания камеры
        self.fullscreen_btn = QPushButton(self.camera_label)
        self.fullscreen_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0,0,0,0.5); 
                border: none;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: rgba(100,100,100,0.7);
            }
        """)
        self.fullscreen_btn.setFixedSize(20, 20)
        self.fullscreen_btn.move(self.camera_label.width() - 25, 5)
        self.fullscreen_btn.clicked.connect(self.toggle_camera_size)
        self.fullscreen_btn.hide()
        
        # Кнопка "Start Game"
        self.start_game_btn = QPushButton("START GAME", self.target_widget)
        self.start_game_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 25px 50px;
                font-size: 28px;
                font-weight: bold;
                border-radius: 10px;
                border: 2px solid #2ecc71;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.start_game_btn.setFixedSize(300, 100)
        self.start_game_btn.clicked.connect(self.start_game)
        
        # Информация о игре
        self.game_score_label = QLabel("SCORE: 0", self.target_widget)
        self.game_score_label.setStyleSheet("""
            font-size: 24px; 
            color: white; 
            background-color: rgba(0, 0, 0, 150);
            padding: 10px;
            border-radius: 5px;
        """)
        self.game_score_label.setAlignment(Qt.AlignCenter)
        self.game_score_label.setFixedSize(200, 50)
        self.game_score_label.hide()
        
        self.game_time_label = QLabel("TIME: 60s", self.target_widget)
        self.game_time_label.setStyleSheet("""
            font-size: 24px; 
            color: white; 
            background-color: rgba(0, 0, 0, 150);
            padding: 10px;
            border-radius: 5px;
        """)
        self.game_time_label.setAlignment(Qt.AlignCenter)
        self.game_time_label.setFixedSize(200, 50)
        self.game_time_label.hide()
        
        # Таймеры
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_frame)
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_game)
        
        # Состояние калибровки
        self.calibration_steps = [
            ("Center", "center"),
            ("Top Left", "top_left"),
            ("Top Right", "top_right"),
            ("Bottom Right", "bottom_right"),
            ("Bottom Left", "bottom_left")
        ]
        self.current_calibration_step = 0
    
    def resizeEvent(self, event):
        if hasattr(self, 'start_game_btn') and self.start_game_btn:
            self.start_game_btn.move(
                self.target_widget.width() // 2 - 150,
                self.target_widget.height() // 2 - 50
            )
        
        if hasattr(self, 'game_score_label') and self.game_score_label:
            self.game_score_label.move(self.target_widget.width() - 220, 20)
            self.game_time_label.move(self.target_widget.width() - 220, 80)
        
        if hasattr(self, 'camera_label') and self.camera_label:
            if self.camera_fullscreen:
                self.camera_label.setFixedSize(self.target_widget.width(), self.target_widget.height())
                self.camera_label.move(0, 0)
            else:
                self.camera_label.setFixedSize(240, 180)
                self.camera_label.move(20, 20)
            
            # Обновляем позицию кнопки разворачивания
            self.fullscreen_btn.move(self.camera_label.width() - 25, 5)
        
        super().resizeEvent(event)
    
    def update_presets_combo(self):
        self.preset_combo.clear()
        for preset in self.target_widget.presets:
            self.preset_combo.addItem(preset['name'])
        self.preset_combo.setCurrentIndex(self.target_widget.current_preset)
        self.delete_preset_btn.setEnabled(self.target_widget.current_preset >= 2)
    
    def create_new_preset(self):
        name, ok = QInputDialog.getText(self, "New Preset", "Enter preset name:")
        if ok and name:
            current_preset = self.target_widget.presets[self.target_widget.current_preset]
            new_index = self.target_widget.add_preset(
                name,
                current_preset["bg_image"],
                current_preset["bg_color"]
            )
            self.target_widget.apply_preset(new_index)
            self.update_presets_combo()
    
    def change_preset(self, index):
        self.target_widget.apply_preset(index)
        self.delete_preset_btn.setEnabled(index >= 2)
        self.target_widget.selected_target = -1
        self.delete_target_btn.setEnabled(False)
    
    def delete_current_preset(self):
        index = self.target_widget.current_preset
        if index < 2:
            QMessageBox.warning(self, "Cannot Delete", "Standard presets cannot be deleted")
            return
            
        reply = QMessageBox.question(self, "Delete Preset", 
                                    f"Are you sure you want to delete preset '{self.target_widget.presets[index]['name']}'?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            del self.target_widget.presets[index]
            new_index = 0 if index == 0 else index - 1
            self.target_widget.apply_preset(new_index)
            self.update_presets_combo()
    
    def toggle_side_panel(self):
        if self.side_panel.width() > 50:
            self.side_panel.setFixedWidth(40)
            self.toggle_btn.setText("▶")
        else:
            self.side_panel.setFixedWidth(300)
            self.toggle_btn.setText("◀")
    
    def toggle_mask_view(self, checked):
        self.show_mask = checked
        if checked:
            self.toggle_mask_btn.setText("Show Camera")
        else:
            self.toggle_mask_btn.setText("Show Mask")
    
    def toggle_camera(self, checked):
        if checked:
            self.camera_toggle.setText("Disable Camera")
            self.camera_active = True
            if not self.laser_detector:
                try:
                    self.laser_detector = GaussianLaserDetector(camera_index=0)
                except RuntimeError as e:
                    QMessageBox.critical(self, "Camera Error", str(e))
                    self.camera_toggle.setChecked(False)
                    return
                
                self.laser_detector.min_brightness = self.brightness_slider.value()
                self.laser_detector.min_red_ratio = self.red_ratio_slider.value() / 100.0
                self.laser_detector.min_gaussian_score = self.gauss_score_slider.value() / 100.0
                self.laser_detector.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_slider.value())
                
                try:
                    if os.path.exists('calibration.json'):
                        with open('calibration.json', 'r') as f:
                            calibration_data = json.load(f)
                        self.laser_detector.load_calibration(calibration_data)
                        self.calibration_status.setText("Calibration: Loaded")
                        
                        # Обновляем поля ввода коррекции
                        dx, dy = self.laser_detector.position_correction
                        self.correction_x.setText(str(dx))
                        self.correction_y.setText(str(dy))
                except:
                    pass
                    
            self.camera_timer.start(30)
            if not self.hide_camera_check.isChecked():
                self.camera_label.show()
                self.fullscreen_btn.show()
        else:
            self.camera_toggle.setText("Enable Camera")
            self.camera_active = False
            self.camera_timer.stop()
            self.camera_label.hide()
            self.fullscreen_btn.hide()
            if self.laser_detector:
                calibration_data = self.laser_detector.save_calibration()
                with open('calibration.json', 'w') as f:
                    json.dump(calibration_data, f)
                    
                self.laser_detector.release()
                self.laser_detector = None
    
    def toggle_camera_visibility(self, state):
        if state == Qt.Checked:
            self.camera_label.hide()
            self.fullscreen_btn.hide()
        else:
            if self.camera_active:
                self.camera_label.show()
                self.fullscreen_btn.show()
    
    def toggle_camera_size(self):
        if self.camera_fullscreen:
            # Возвращаем к обычному размеру
            self.camera_label.setFixedSize(240, 180)
            self.camera_label.move(20, 20)
            self.fullscreen_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_TitleBarMaxButton))
            self.camera_fullscreen = False
        else:
            # Разворачиваем на весь экран
            self.camera_label.setFixedSize(self.target_widget.width(), self.target_widget.height())
            self.camera_label.move(0, 0)
            self.fullscreen_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_TitleBarNormalButton))
            self.camera_fullscreen = True
        
        # Обновляем позицию кнопки
        self.fullscreen_btn.move(self.camera_label.width() - 25, 5)
    
    def start_calibration(self):
        if not self.laser_detector or not self.camera_active:
            return
            
        self.current_calibration_step = 0
        self.next_step_btn.setEnabled(True)
        self.calibration_btn.setEnabled(False)
        self.next_calibration_step()
    
    def next_calibration_step(self):
        if self.current_calibration_step >= len(self.calibration_steps):
            # Завершаем калибровку
            self.laser_detector.finish_calibration()
            self.calibration_status.setText("Calibration: Completed")
            self.calibration_btn.setEnabled(True)
            self.next_step_btn.setEnabled(False)
            self.target_widget.calibration_point = None
            QMessageBox.information(self, "Calibration", "All calibration points saved successfully!")
            return
            
        step_name, target_name = self.calibration_steps[self.current_calibration_step]
        self.laser_detector.start_calibration(target_name)
        self.calibration_status.setText(f"Calibration: Point at {step_name}")
        
        # Устанавливаем визуальную метку для текущей точки калибровки
        if step_name == "Center":
            pos = QPoint(self.target_widget.width() // 2, self.target_widget.height() // 2)
        elif step_name == "Top Left":
            pos = QPoint(50, 50)
        elif step_name == "Top Right":
            pos = QPoint(self.target_widget.width() - 50, 50)
        elif step_name == "Bottom Right":
            pos = QPoint(self.target_widget.width() - 50, self.target_widget.height() - 50)
        elif step_name == "Bottom Left":
            pos = QPoint(50, self.target_widget.height() - 50)
        
        self.target_widget.set_calibration_point(step_name, pos)
        self.current_calibration_step += 1
    
    def reset_calibration(self):
        """Сброс калибровки до начальных значений"""
        if self.laser_detector:
            self.laser_detector.reset_calibration()
            self.calibration_status.setText("Calibration: Reset")
            self.current_calibration_step = 0
            self.calibration_btn.setEnabled(True)
            self.next_step_btn.setEnabled(False)
            self.target_widget.calibration_point = None
            
            # Обновляем поля ввода коррекции
            self.correction_x.setText("0")
            self.correction_y.setText("0")
            
            # Очищаем файл калибровки
            if os.path.exists('calibration.json'):
                try:
                    os.remove('calibration.json')
                except:
                    pass
    
    def toggle_correction_mode(self, checked):
        if not self.laser_detector or not self.camera_active:
            self.correction_btn.setChecked(False)
            return
            
        self.target_widget.correction_mode = checked
        if checked:
            self.target_widget.correction_point = None
            self.target_widget.correction_offset = (0, 0)
            self.correction_btn.setText("Finish Correction")
            
            # Обновляем поля ввода текущими значениями
            dx, dy = self.laser_detector.position_correction
            self.correction_x.setText(str(dx))
            self.correction_y.setText(str(dy))
        else:
            # Применяем коррекцию
            if self.target_widget.correction_offset != (0, 0):
                self.laser_detector.set_position_correction(
                    self.target_widget.correction_offset[0],
                    self.target_widget.correction_offset[1]
                )
                self.calibration_status.setText("Position correction applied")
                
                # Обновляем поля ввода
                dx, dy = self.laser_detector.position_correction
                self.correction_x.setText(str(dx))
                self.correction_y.setText(str(dy))
            self.correction_btn.setText("Position Correction")
    
    def update_correction(self):
        """Обновляет коррекцию позиции на основе введенных значений"""
        if not self.laser_detector:
            return
            
        try:
            dx = int(self.correction_x.text())
        except:
            dx = 0
            
        try:
            dy = int(self.correction_y.text())
        except:
            dy = 0
            
        self.laser_detector.set_position_correction(dx, dy)
        self.calibration_status.setText(f"Position correction: X={dx}, Y={dy}")
    
    def set_brightness(self, value):
        if self.laser_detector and self.camera_active:
            self.laser_detector.min_brightness = value
    
    def set_red_ratio(self, value):
        if self.laser_detector and self.camera_active:
            self.laser_detector.min_red_ratio = value / 100.0
    
    def set_gauss_score(self, value):
        if self.laser_detector and self.camera_active:
            self.laser_detector.min_gaussian_score = value / 100.0
    
    def set_exposure(self, value):
        if self.laser_detector and self.camera_active:
            self.laser_detector.cap.set(cv2.CAP_PROP_EXPOSURE, value)
    
    def select_background(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "", "Image Files (*.png *.jpg *.bmp)"
        )
        if file_name:
            self.target_widget.set_background(file_name)
    
    def add_new_target(self, shape):
        self.target_widget.add_target(shape)
        # Выбираем новую мишень
        self.target_widget.selected_target = len(self.target_widget.targets) - 1
        self.delete_target_btn.setEnabled(True)
        # Устанавливаем размер на слайдере
        self.target_size_slider.setValue(self.target_widget.targets[-1]['size'])
    
    def delete_selected_target(self):
        if self.target_widget.selected_target >= 0:
            self.target_widget.remove_target(self.target_widget.selected_target)
            self.delete_target_btn.setEnabled(self.target_widget.selected_target >= 0)
    
    def update_target_size(self, size):
        if self.target_widget.selected_target >= 0:
            self.target_widget.targets[self.target_widget.selected_target]['size'] = size
            self.target_widget.update()
            if self.target_widget.current_preset >= 0:
                self.target_widget.presets[self.target_widget.current_preset]["targets"] = [t.copy() for t in self.target_widget.targets]
    
    def update_frame(self):
        if self.laser_detector:
            result = self.laser_detector.process_frame()
            if result:
                frame, laser_pos, _ = result
                
                # Преобразуем координаты с помощью гомографии
                if not self.laser_detector.calibration_mode and self.laser_detector.homography_matrix is not None:
                    laser_pos = self.laser_detector.transform_point(laser_pos)
                
                # Отображаем кадр в мини-окне камеры
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(
                    self.camera_label.width(), 
                    self.camera_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.camera_label.setPixmap(QPixmap.fromImage(p))
                
                # Обновляем позицию лазера
                if laser_pos:
                    self.target_widget.laser_detected = True
                    self.target_widget.set_laser_position(laser_pos)
                    
                    # Проверка попадания
                    if self.game_active and not self.hit_cooldown:
                        self.check_hit(laser_pos)
                else:
                    self.target_widget.laser_detected = False
    
    def start_game(self):
        if not self.game_active:
            self.start_game_btn.hide()
            self.game_score_label.show()
            self.game_time_label.show()
            self.side_panel.setFixedWidth(40)
            self.toggle_btn.setText("▶")
            self.score = 0
            self.game_time = 60
            self.game_score_label.setText(f"SCORE: {self.score}")
            self.game_time_label.setText(f"TIME: {self.game_time}s")
            self.start_btn.setText("Stop Game")
            self.game_active = True
            self.game_timer.start(1000)
            
            if not self.camera_active:
                self.camera_toggle.setChecked(True)
                self.toggle_camera(True)
        else:
            self.stop_game()
    
    def stop_game(self):
        self.game_active = False
        self.game_timer.stop()
        self.start_btn.setText("Start Game")
        self.start_game_btn.show()
        self.game_score_label.hide()
        self.game_time_label.hide()
        self.side_panel.setFixedWidth(300)
        self.toggle_btn.setText("◀")
        
        if self.parent.current_user:
            self.parent.save_score(self.parent.current_user, self.score)
    
    def update_game(self):
        self.game_time -= 1
        self.game_time_label.setText(f"TIME: {self.game_time}s")
        
        if self.game_time <= 0:
            self.stop_game()
            QMessageBox.information(self, "Game Over", f"Game Over!\nYour Score: {self.score}")
    
    def check_hit(self, laser_pos):
        for target in self.target_widget.targets:
            target_x = target['pos'].x()
            target_y = target['pos'].y()
            target_size = target['size']
            
            if target_size <= 0:
                continue
                
            distance = np.sqrt((laser_pos[0] - target_x)**2 + (laser_pos[1] - target_y)**2)
            
            if target['shape'] == TargetWidget.CIRCLE:
                if distance < target_size:
                    zone = int(distance / (target_size / 5))
                    points = 10 - zone * 2
                    
                    if points < 2:
                        points = 2
                    
                    self.score += points
                    self.game_score_label.setText(f"SCORE: {self.score}")
                    self.target_widget.add_hit(laser_pos, points)
                    self.hit_cooldown = True
                    QTimer.singleShot(300, lambda: setattr(self, 'hit_cooldown', False))
                    break
            
            elif target['shape'] == TargetWidget.RECTANGLE:
                dx = abs(laser_pos[0] - target_x)
                dy = abs(laser_pos[1] - target_y)
                
                if dx < target_size and dy < target_size:
                    norm_dist = max(dx, dy) / target_size
                    zone = int(norm_dist * 5)
                    points = 10 - zone
                    
                    if points < 1:
                        points = 1
                    
                    self.score += points
                    self.game_score_label.setText(f"SCORE: {self.score}")
                    self.target_widget.add_hit(laser_pos, points)
                    self.hit_cooldown = True
                    QTimer.singleShot(300, lambda: setattr(self, 'hit_cooldown', False))
                    break
            
            elif target['shape'] == TargetWidget.DIAMOND:
                # Для ромба используем метрику Манхэттена
                manhattan_dist = abs(laser_pos[0] - target_x) + abs(laser_pos[1] - target_y)
                if manhattan_dist < target_size:
                    zone = int(manhattan_dist / (target_size / 5))
                    points = 10 - zone * 2
                    
                    if points < 2:
                        points = 2
                    
                    self.score += points
                    self.game_score_label.setText(f"SCORE: {self.score}")
                    self.target_widget.add_hit(laser_pos, points)
                    self.hit_cooldown = True
                    QTimer.singleShot(300, lambda: setattr(self, 'hit_cooldown', False))
                    break
    
    def back_to_menu(self):
        if self.game_active:
            self.stop_game()
            
        if self.laser_detector:
            calibration_data = self.laser_detector.save_calibration()
            with open('calibration.json', 'w') as f:
                json.dump(calibration_data, f)
                
            self.laser_detector.release()
            self.laser_detector = None
            
        self.camera_timer.stop()
        self.parent.stacked_widget.setCurrentIndex(0)

# Окно настроек
class SettingsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(layout)
        
        title = QLabel("SETTINGS")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #3498db;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        camera_group = QGroupBox("CAMERA SETTINGS")
        camera_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #3498db;
            }
        """)
        camera_layout = QVBoxLayout()
        
        available_cameras = find_available_cameras()
        if not available_cameras:
            available_cameras = [0]
        
        camera_label = QLabel("Select Camera:")
        camera_label.setStyleSheet("color: #ecf0f1;")
        camera_layout.addWidget(camera_label)
        
        self.camera_combo = QComboBox()
        for cam_index in available_cameras:
            self.camera_combo.addItem(f"Camera {cam_index}", cam_index)
        camera_layout.addWidget(self.camera_combo)
        
        resolution_label = QLabel("Resolution:")
        resolution_label.setStyleSheet("color: #ecf0f1;")
        camera_layout.addWidget(resolution_label)
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.resolution_combo.setCurrentIndex(1)
        camera_layout.addWidget(self.resolution_combo)
        
        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet("color: #ecf0f1;")
        camera_layout.addWidget(fps_label)
        
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(5, 60)
        self.fps_slider.setValue(30)
        camera_layout.addWidget(self.fps_slider)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        back_btn = QPushButton("Back to Menu")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                padding: 10px;
                margin-top: 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #95a5a6;
            }
        """)
        back_btn.clicked.connect(self.back_to_menu)
        layout.addWidget(back_btn)
    
    def get_selected_camera_index(self):
        return self.camera_combo.currentData()
    
    def back_to_menu(self):
        self.parent.stacked_widget.setCurrentIndex(0)

# Окно рекордов
class RecordsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        self.load_records()
    
    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(layout)
        
        title = QLabel("HIGH SCORES")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #e74c3c;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Rank", "Player", "Score", "Date"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #ecf0f1;
                gridline-color: #bdc3c7;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                font-weight: bold;
                padding: 5px;
            }
        """)
        layout.addWidget(self.table)
        
        back_btn = QPushButton("Back to Menu")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                padding: 10px;
                margin-top: 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #95a5a6;
            }
        """)
        back_btn.clicked.connect(self.back_to_menu)
        layout.addWidget(back_btn)
    
    def load_records(self):
        try:
            if os.path.exists("records.json"):
                with open("records.json", "r") as f:
                    records = json.load(f)
                
                sorted_records = sorted(records, key=lambda x: x["score"], reverse=True)
                top_records = sorted_records[:10]
                
                self.table.setRowCount(len(top_records))
                for i, record in enumerate(top_records):
                    self.table.setItem(i, 0, QTableWidgetItem(str(i+1)))
                    self.table.setItem(i, 1, QTableWidgetItem(record["player"]))
                    self.table.setItem(i, 2, QTableWidgetItem(str(record["score"])))
                    self.table.setItem(i, 3, QTableWidgetItem(record["date"]))
        except:
            pass
    
    def back_to_menu(self):
        self.parent.stacked_widget.setCurrentIndex(0)

# Главное окно приложения
class LaserTaskApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lazer Task")
        self.setGeometry(100, 100, 1200, 800)
        self.current_user = None
        self.show_login()
    
    def show_login(self):
        login_dialog = LoginDialog(self)
        if login_dialog.exec_() == QDialog.Accepted:
            self.current_user = login_dialog.current_user
            self.init_main_app()
        else:
            QApplication.quit()
    
    def init_main_app(self):
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.main_menu = MainMenu(self)
        self.game_window = GameWindow(self)
        self.settings_window = SettingsWindow(self)
        self.records_window = RecordsWindow(self)
        
        self.stacked_widget.addWidget(self.main_menu)
        self.stacked_widget.addWidget(self.game_window)
        self.stacked_widget.addWidget(self.settings_window)
        self.stacked_widget.addWidget(self.records_window)
        
        self.stacked_widget.setCurrentIndex(0)
    
    def save_score(self, player, score):
        try:
            if os.path.exists("records.json"):
                with open("records.json", "r") as f:
                    records = json.load(f)
            else:
                records = []
            
            records.append({
                "player": player,
                "score": score,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            
            with open("records.json", "w") as f:
                json.dump(records, f)
        except:
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    dark_palette = app.palette()
    dark_palette.setColor(dark_palette.Window, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.WindowText, Qt.white)
    dark_palette.setColor(dark_palette.Base, QColor(35, 35, 35))
    dark_palette.setColor(dark_palette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ToolTipBase, Qt.white)
    dark_palette.setColor(dark_palette.ToolTipText, Qt.white)
    dark_palette.setColor(dark_palette.Text, Qt.white)
    dark_palette.setColor(dark_palette.Button, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ButtonText, Qt.white)
    dark_palette.setColor(dark_palette.BrightText, Qt.red)
    dark_palette.setColor(dark_palette.Highlight, QColor(142, 45, 197).lighter())
    dark_palette.setColor(dark_palette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    font = QFont("Arial", 10)
    app.setFont(font)
    
    main_win = LaserTaskApp()
    main_win.show()
    sys.exit(app.exec_())