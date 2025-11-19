#Se necesita de un entorno conda para ejecutar el programa
import cv2
import mediapipe as mp #version 4.9.0.80
import time

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
# Inicializa la utilidad de dibujo de MediaPipe
mp_drawing = mp.solutions.drawing_utils
# Estilo de dibujo para los puntos clave
drawing_styles = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2) # Verde

# Abre la webcam (0 es el índice de la cámara principal)
cap = cv2.VideoCapture(0)

# Verificación de apertura de cámara
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Cámara abierta. Presiona 'q' para salir.")

# Bucle principal para leer fotogramas de la cámara
while cap.isOpened():
    # Leer el fotograma (frame)
    success, image = cap.read()

    if not success:
        print("Ignorando fotograma de la cámara.")
        continue

    # 1. Procesamiento de la imagen para MediaPipe
    # MediaPipe requiere imágenes RGB. OpenCV lee BGR.
    image.flags.writeable = False # Optimización
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Procesar con MediaPipe
    results = hands.process(image)

    # 2. Devolver la imagen a BGR para mostrar con OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 3. Dibujar los puntos clave (landmarks) si se detecta una mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibuja los puntos y las conexiones de la mano
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles,  # Estilo para los puntos
                drawing_styles   # Estilo para las conexiones
            )

    # Mostrar el fotograma en una ventana
    cv2.imshow('MediaPipe Hand Detection | Pulsa Q para salir', image)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()