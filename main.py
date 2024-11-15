# Importación de Librerías
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

# Inicialización de la librería MediaPipe para detección de rostros
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

# Inicialización de la malla facial de MediaPipe
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

# Realizar nuestra video captura
cap = cv2.VideoCapture(0)


# Función para obtener los frames de la cámara
def gen_frame():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        else:
            # Conversión de BGR a RGB
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesamiento de los resultados de la malla facial
            resultados = MallaFacial.process(frameRGB)

            # Si se detectan rostros
            if resultados.multi_face_landmarks:
                for rostros in resultados.multi_face_landmarks:
                    # Dibujo de la malla facial
                    mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

            # Codificación de la imagen en formato JPEG
            suc, encode = cv2.imencode('.jpg', frame)
            frame = encode.tobytes()

            # Enviar el frame como un flujo de video
        yield(b'--frame\r\n'
              b'content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Crear la aplicación Flask
app = Flask(__name__)


# Ruta principal
@app.route('/')
def index():
    return render_template('Index.html')


# Ruta de video
@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Ejecutar la web
if __name__ == "__main__":
    app.run(debug=True)
    cap.release()
    cv2.destroyAllWindows()

