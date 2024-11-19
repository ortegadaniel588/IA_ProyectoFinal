# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:05:55 2024

@author: Camilo
"""

# Importar las bibliotecas necesarias
import cv2

# Cargar el clasificador Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video desde la cámara (0 significa la cámara por defecto del sistema)
cap = cv2.VideoCapture(0)

while True:
    # Capturar fotograma por fotograma desde la cámara
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el fotograma
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar el fotograma con los rostros detectados
    cv2.imshow('Detección de rostros en tiempo real', frame)

    # Salir del bucle al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura y cerrar todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
