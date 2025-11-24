import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# --- INICIALIZACIÓN Y CONFIGURACIÓN ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Archivos de poses
POSE_OPEN_FILE = 'mano_abierta_pose.pkl'
POSE_CLOSED_FILE = 'mano_cerrada_pose.pkl'

# Variables para guardar las poses
saved_pose_open = None
saved_pose_closed = None

# Umbral de similitud global (ajustar si es necesario)
THRESHOLD = 0.8

# --- Cargar Poses Guardadas ---

def load_pose(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            print(f"Pose '{file_path}' cargada exitosamente.")
            return pickle.load(f)
    return None

saved_pose_open = load_pose(POSE_OPEN_FILE)
saved_pose_closed = load_pose(POSE_CLOSED_FILE)

# --- Cargar Poses Guardadas ---
def load_pose(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            print(f"Pose '{file_path}' cargada exitosamente.")
            return pickle.load(f)
    return None

# --- FUNCIONES DE PROCESAMIENTO ---
def extract_normalized_pose_stable(hand_landmarks):
    """
    Convierte los 21 landmarks a una matriz 2D (X, Y), 
    normalizada por traslación, escala Y ROTACIÓN.
    """
    # Usar solo X e Y (el eje Z es muy ruidoso para la normalización de rotación)
    landmarks = np.array([[lmk.x, lmk.y] for lmk in hand_landmarks.landmark]) 
    
    # 1. TRASLACIÓN (Mover la muñeca (0) al origen (0, 0))
    muñeca = landmarks[0]
    landmarks_relativos = landmarks - muñeca
    
    # 2. ESCALADO (Usar la distancia de la muñeca (0) al nudillo mayor (5) como escala)
    escala_vector = landmarks_relativos[5]
    distancia_escala = np.linalg.norm(escala_vector)
    
    if distancia_escala == 0:
        return None 
    
    landmarks_escalados = landmarks_relativos / distancia_escala
    
    # 3. NORMALIZACIÓN DE ROTACIÓN (Alinear el vector (0 -> 5) con el eje X)
    
    # Coordenadas relativas del punto 5 (ya escaladas)
    p5_x, p5_y = landmarks_escalados[5] 
    
    # Calcular el ángulo de rotación (ángulo que el vector (0->5) hace con el eje X)
    angle = np.arctan2(p5_y, p5_x) 
    
    # Crear matriz de rotación para girar por el ángulo negativo
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], 
                  [s, c]])
    
    # Aplicar la rotación a todos los landmarks escalados
    landmarks_rotados = np.dot(landmarks_escalados, R.T)
    
    # Aplanar a 1D (21 * 2 = 42 valores)
    return landmarks_rotados.flatten()
def compare_poses(pose1, pose2):
    """Calcula la diferencia (distancia euclidiana) entre dos poses."""
    return np.linalg.norm(pose1 - pose2)


# --- BUCLE PRINCIPAL ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Preprocesamiento
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Texto de estado inicial
    display_text = "Esperando mano..."
    color = (255, 255, 255) # Blanco

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            current_pose = extract_normalized_pose_stable(hand_landmarks)
            
            if current_pose is None:
                continue

            # --- Lógica de COMPARACIÓN ---
            
            # Verificar si ambas poses están guardadas
            if saved_pose_open is not None and saved_pose_closed is not None:
                dist_open = compare_poses(saved_pose_open, current_pose)
                dist_closed = compare_poses(saved_pose_closed, current_pose)
                
                # Se clasifica como la pose a la que tenga menor distancia
                if dist_open < dist_closed and dist_open < THRESHOLD:
                    display_text = f"GESTO: MANO ABIERTA (Dist: {dist_open:.3f})"
                    color = (0, 255, 0) # Verde
                elif dist_closed < dist_open and dist_closed < THRESHOLD:
                    display_text = f"GESTO: MANO CERRADA (Dist: {dist_closed:.3f})"
                    color = (255, 0, 0) # Rojo
                else:
                    display_text = f"Pose no reconocida (Dif. Abierta: {dist_open:.3f}, Cerrada: {dist_closed:.3f})"
                    color = (255, 255, 0) # Amarillo
            
            # Mensaje de configuración si faltan poses
            elif saved_pose_open is None:
                display_text = "CONFIGURACIÓN: Presiona 's' para guardar la MANO ABIERTA."
                color = (0, 165, 255) # Naranja
            elif saved_pose_closed is None:
                display_text = "CONFIGURACIÓN: Presiona 'c' para guardar la MANO CERRADA (Puño)."
                color = (0, 165, 255) # Naranja


    # Mostrar estado en la imagen
    cv2.putText(image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imshow('Deteccion de Gestos (S=Abrir, C=Cerrar, Q=Salir)', image)
    
    key = cv2.waitKey(5) & 0xFF
    
    # --- LÓGICA CLAVE: Guardar Poses (Teclas 's' y 'c') ---
    if results.multi_hand_landmarks:
        hand_landmarks_to_save = results.multi_hand_landmarks[0]
        pose_to_save = extract_normalized_pose_stable(hand_landmarks_to_save)
        
        # Guardar MANO ABIERTA (tecla 's')
        if key == ord('s'):
            saved_pose_open = pose_to_save
            if saved_pose_open is not None:
                with open(POSE_OPEN_FILE, 'wb') as f:
                    pickle.dump(saved_pose_open, f)
                print(f"Pose 'MANO ABIERTA' GUARDADA en '{POSE_OPEN_FILE}'.")
                display_text = "¡MANO ABIERTA GUARDADA!"
        
        # Guardar MANO CERRADA (tecla 'c')
        if key == ord('c'):
            saved_pose_closed = pose_to_save
            if saved_pose_closed is not None:
                with open(POSE_CLOSED_FILE, 'wb') as f:
                    pickle.dump(saved_pose_closed, f)
                print(f"Pose 'MANO CERRADA' GUARDADA en '{POSE_CLOSED_FILE}'.")
                display_text = "¡MANO CERRADA GUARDADA!"

    # Salir del bucle (tecla 'q')
    if key == ord('q'):
        break

# Cierre
cap.release()
cv2.destroyAllWindows()