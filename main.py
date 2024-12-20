# Importación de Librerías
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os

# Cargar imágenes de referencia para reconocimiento facial
known_face_encodings = []
known_face_names = []

# Carpeta con imágenes de personas conocidas
path_to_images = "known_faces"  # Cambia esta ruta según tus archivos
for filename in os.listdir(path_to_images):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(path_to_images, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]  # Asegúrate de que hay un rostro por imagen
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])  # Usa el nombre del archivo como etiqueta

# Inicialización de captura de video
cap = cv2.VideoCapture(0)

# Función para obtener los frames de la cámara
def gen_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB para usar face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detección de las ubicaciones de los rostros
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Comparar el rostro actual con los rostros conocidos
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Acceso denegado"  # Por defecto, desconocido

            # Calcular las distancias y encontrar el mejor emparejamiento
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Elegir el color basado en si el rostro es conocido o no
            if name == "Acceso denegado":
                color = (0, 0, 255)  # Rojo para desconocidos
            else:
                color = (0, 255, 0)  # Verde para conocidos

            # Dibujar el rectángulo alrededor del rostro
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)

            # Poner el texto encima del rectángulo
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Codificar la imagen en formato JPEG
        suc, encode = cv2.imencode('.jpg', frame)
        frame = encode.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta principal
@app.route('/')
def index():
    return render_template('AI-html-1.0.0/index.html')

# Ruta de video
@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ejecutar la web
if __name__ == "__main__":
    app.run(debug=True)
    cap.release()
    cv2.destroyAllWindows()
