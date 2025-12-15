# Nel file detect_face.py - SOSTITUISCI dlib con OpenCV Haar Cascades

import cv2

class FaceDetector:
    def __init__(self):
        # Carica il classificatore Haar per volti (incluso in OpenCV)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Opzionale: per occhi
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, image):
        """
        Rileva volti usando Haar Cascades
        
        Args:
            image: numpy array BGR (OpenCV format)
        
        Returns:
            List di bounding boxes [(x, y, w, h), ...]
        """
        # Converti in scala di grigi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Rileva volti
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces  # Già nel formato [(x, y, w, h), ...]
    
    def detect_faces_with_eyes(self, image):
        """Rileva volti e verifica presenza occhi per maggiore accuratezza"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        validated_faces = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Accetta solo volti con almeno 2 occhi rilevati
            if len(eyes) >= 2:
                validated_faces.append((x, y, w, h))
        
        return validated_faces if validated_faces else faces

# ESEMPIO DI USO:
# detector = FaceDetector()
# faces = detector.detect_faces(image)
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
