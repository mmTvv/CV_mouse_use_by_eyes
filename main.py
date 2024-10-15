import cv2
import dlib
import pyautogui
from scipy.spatial import distance
import numpy as np
import time

# Загрузим предсказатель лицевых меток
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Идентификация ключевых точек для глаз
(left_eye_start, left_eye_end) = (36, 42)
(right_eye_start, right_eye_end) = (42, 48)

# Функция для вычисления коэффициента открытия глаза (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Функция для нахождения центра глаза
def get_eye_center(eye):
    x_coords = [p[0] for p in eye]
    y_coords = [p[1] for p in eye]
    center_x = int(sum(x_coords) / len(x_coords))
    center_y = int(sum(y_coords) / len(y_coords))
    return (center_x, center_y)

# Порог для морганий и число морганий для двойного клика
BLINK_THRESHOLD = 0.33  # Более чувствительный порог моргания
BLINK_CONSEC_FRAMES = 1  # Более чувствительное срабатывание по количеству кадров
DOUBLE_BLINK_TIME = 1.0  # Время между морганиями для двойного клика

blink_counter = 0
total_blinks = 0
last_blink_time = 0

# Размер экрана
screen_width, screen_height = pyautogui.size()

# Скорость перемещения мыши
MOVE_SPEED = 2  # Параметр, определяющий скорость движения мыши в пикселях

# Подключаем камеру
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Преобразуем изображение в серый формат
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Определяем лицо на изображении
    faces = detector(gray, 0)
    
    for face in faces:
        # Определяем ключевые точки на лице
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        # Получаем координаты левого и правого глаза
        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]
        
        # Получаем центры глаз
        left_eye_center = get_eye_center(left_eye)
        right_eye_center = get_eye_center(right_eye)
        
        # Определяем смещение зрачков от центра кадра (влево/вправо, вверх/вниз)
        gaze_x = (left_eye_center[0] + right_eye_center[0]) // 2
        gaze_y = (left_eye_center[1] + right_eye_center[1]) // 2
        
        # Рассчитываем центр кадра (это будет наша базовая точка)
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        
        # Вычисляем отклонение взгляда от центра
        delta_x = frame_center_x - gaze_x  # Инвертируем X
        delta_y = gaze_y - frame_center_y  # Y оставляем как есть
        
        # Двигаем мышь в направлении взгляда (умножаем на MOVE_SPEED для ускорения)
        pyautogui.moveRel(delta_x // MOVE_SPEED, delta_y // MOVE_SPEED, duration=0.1)
        
        # Вычисляем коэффициент открытия глаза (EAR)
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Проверяем, закрыты ли глаза (моргание)
        if ear < BLINK_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_CONSEC_FRAMES:
                total_blinks += 1
                current_time = time.time()
                
                # Проверка на двойное моргание
                if current_time - last_blink_time < DOUBLE_BLINK_TIME:
                    pyautogui.click()  # Двойной клик
                    print("Double blink detected, mouse click performed")
                
                last_blink_time = current_time
            
            blink_counter = 0
        
        # Рисуем контуры глаз для визуализации
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    # Показываем изображение
    cv2.imshow("Eye Tracking", frame)
    
    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
