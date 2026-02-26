from deepface import DeepFace
import os
import cv2
import numpy as np
import base64

class DeepFaceService:
    def __init__(self):
        self.model = "Facenet"
        self.detector = "retinaface" # ou 'retinaface' pour plus de précision

    def process_image(self, img_path, enforce=True): # <--- BIEN VÉRIFIER CETTE LIGNE
        # On choisit le détecteur selon l'importance (retinaface est plus costaud)
        detector_to_use = "retinaface" if enforce else "ssd"
        
        results = DeepFace.represent(
            img_path=img_path, 
            model_name=self.model, 
            detector_backend=detector_to_use,
            enforce_detection=enforce,
            align=True
        )
        
        if not results:
            return None
            
        return {
            "embedding": results[0]["embedding"],
            "confidence": results[0].get("face_confidence", 0)
        }

    def analyze_face(self, img_path):
        # Fonction synchrone pour l'analyse
        results = DeepFace.analyze(
            img_path=img_path, 
            actions=['age', 'gender', 'race', 'emotion'],
            detector_backend=self.detector,
            enforce_detection=False # <--- AJOUTE ÇA AUSSI
        )
        return results[0]
    
    def verify_faces(self, img1_path, img2_path):
        # Utilise la méthode native de DeepFace mentionnée dans le README
        result = DeepFace.verify(
            img1_path = img1_path, 
            img2_path = img2_path, 
            model_name = self.model,
            detector_backend = self.detector,
            enforce_detection = True # Erreur si aucun visage n'est trouvé
        )
        return result
    def check_image_quality(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return False, "Erreur de lecture image"

        # 1. Vérification de la luminosité
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        if brightness < 40:
            return False, f"Trop sombre ({int(brightness)})"
        if brightness > 220:
            return False, f"Trop de lumière ({int(brightness)})"

        # 2. Vérification du flou (Laplacian)
        # Plus le score est bas, plus c'est flou. 
        # Un score < 100 est souvent signe d'un mauvais focus.
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:
            return False, f"Image trop floue ({int(blur_score)})"
        # On élargit la luminosité (30 au lieu de 40)
        if brightness < 30:
            return False, "Trop sombre"

        return True, "Qualité OK"
        
dp_service = DeepFaceService()