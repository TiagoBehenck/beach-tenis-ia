import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import threading
import queue
import math

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class BeachTenisIA: 
  def __init__(self, file_name='example.mp4'):
      self.file_name = file_name
      self.image_q = queue.Queue()
      model_path = 'pose_landmarker_full.task'

      self.options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
      )
  
  def find_angle(self, frame, landmarks, p1, p2, p3, draw):
    land = landmarks.pose_landmarks[0]
    h, w, c = frame.shape
    x1, y1 = (land[p1].x, land[p1].y)
    x2, y2 = (land[p2].x, land[p2].y)
    x3, y3 = (land[p3].x, land[p3].y)

    angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                         math.atan2(y1-y2, x1-x2))
    
    position = (int(x2 * w + 10), int(y2 * h + 10))

    if draw:
       frame = cv2.putText(frame ,str(int(angle)), position, cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    return frame, angle

  def draw_landmarks_on_image(self, rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
      pose_landmarks = pose_landmarks_list[idx]

      # Draw the pose landmarks.
      pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

  def analyze_tennis_serve(self, elbow_angle, previous_angles=None):
    """
    Analisa a biomecânica do saque de tênis baseado nos ângulos do cotovelo
    
    Parâmetros:
    - elbow_angle: Ângulo atual do cotovelo (pode ser objeto ou número)
    - previous_angles: Lista dos últimos ângulos registrados (opcional)
    
    Retorna um dicionário com análise do movimento
    """
    # Função para extrair valor numérico de diferentes tipos
    def extract_angle(angle):
        # Se for um objeto, tenta extrair valor numérico
        if hasattr(angle, 'value'):
            return float(angle.value)
        # Se for um número, converte para float
        return float(angle)
    
    # Se previous_angles não for fornecido, inicializar como lista vazia
    if previous_angles is None:
        previous_angles = []
    
    # Converter ângulos para valores numéricos
    try:
        elbow_angle = extract_angle(elbow_angle)
        previous_angles = [extract_angle(angle) for angle in previous_angles]
    except (TypeError, ValueError) as e:
        print(f"Erro ao converter ângulos: {e}")
        return None
    
    # Definir fases e limites de ângulos
    phases = {
        'preparacao': (130, 160),   # Fase inicial de preparação
        'aceleracao': (60, 130),    # Fase de aceleração do saque
        'contato': (30, 60),        # Momento do contato com a bola
        'seguimento': (130, 170)    # Fase final de seguimento
    }
    
    # Variáveis para análise do movimento
    analysis = {
        'current_phase': None,
        'movement_quality': None,
        'speed_potential': None
    }
    
    # Determinar fase atual baseado no ângulo
    for phase, (min_angle, max_angle) in phases.items():
        if min_angle <= elbow_angle <= max_angle:
            analysis['current_phase'] = phase
            break
    
    # Calcular variação angular (velocidade)
    if len(previous_angles) > 0:
        angle_variation = abs(elbow_angle - previous_angles[-1])
        
        # Avaliar qualidade do movimento baseado na variação angular
        if angle_variation > 20:
            analysis['movement_quality'] = 'Rápido'
            analysis['speed_potential'] = 'Alto'
        elif 10 <= angle_variation <= 20:
            analysis['movement_quality'] = 'Moderado'
            analysis['speed_potential'] = 'Médio'
        else:
            analysis['movement_quality'] = 'Lento'
            analysis['speed_potential'] = 'Baixo'
    
    return analysis
  
  def process_video(self, draw, display):
    with PoseLandmarker.create_from_options(self.options) as landmarker:
      cap = cv2.VideoCapture(self.file_name)
      calc_ts = [0.0]
      
      while (cap.isOpened()):
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)

        if ret == True:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            calc_ts.append(int(calc_ts[-1] + 1000/fps))
            
            detection_result = landmarker.detect_for_video(mp_image, calc_ts[-1])

            if draw: 
              frame = self.draw_landmarks_on_image(frame, detection_result)
            
            if display:
              cv2.imshow('Frame', frame)

            # Press Q to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.image_q.put((frame, detection_result, calc_ts[-1]))
        else: 
            break
    self.image_q.put((1, 1, 'done'))
    cap.release()
    cv2.destroyAllWindows()

  def run(self, draw, display=False):
     t1 = threading.Thread(target=self.process_video, args=(draw, display))
     t1.start()

if __name__ == "__main__":
  beachTenisIA = BeachTenisIA()
  beachTenisIA.process_video(True, True)