from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.services.dp_services import dp_service
from app.core.wea import db
from app.schemas.response import FaceAnalysis, SearchResult, MsgResponse, VerifyResponse, Base64Request
import aiofiles
import os
import base64
import uuid
from typing import List

router = APIRouter()

@router.post("/register", response_model=MsgResponse)
async def register(name: str = Form(...), file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    try:
        # 1. Sauvegarde asynchrone du fichier
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # 2. Extraction embedding (CPU bound -> threadpool)
        embedding = await run_in_threadpool(dp_service.process_image, temp_path)

        # 3. Stockage Weaviate (CORRECTION ICI)
        faces = db.client.collections.get("Face_Facenet")
        
        # On utilise .data.insert
        faces.data.insert(
            properties={"person_name": name},
            vector=embedding
        )
        
        return {"message": f"Profil de {name} créé avec succès."}
    
    except Exception as e:
        print(f"Erreur lors de l'enregistrement : {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



@router.post("/register-multi")
async def register_multi(name: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Aucune image reçue")
    
    faces = db.client.collections.get("Face_Facenet")
    results_count = 0
    errors = []

    for idx, file in enumerate(files):
        temp_path = f"multi_{name}_{idx}_{uuid.uuid4()}.jpg"
        try:
            # 1. Sauvegarde
            content = await file.read()
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(content)

            # 2. Qualité & Extraction (On force RetinaFace pour la base de données)
            # On utilise une version plus permissive de check_quality pour le batch
            is_ok, _ = dp_service.check_image_quality(temp_path)
            
            # Extraction du vecteur
            res = await run_in_threadpool(dp_service.process_image, temp_path, enforce=True)
            
            if res and "embedding" in res:
                # 3. Insertion dans le nuage
                faces.data.insert(
                    properties={
                        "person_name": name,
                        "pose_index": idx
                    },
                    vector=res["embedding"]
                )
                results_count += 1
        except Exception as e:
            errors.append(f"Image {idx}: {str(e)}")
            continue
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if results_count == 0:
        raise HTTPException(status_code=400, detail=f"Échec total de l'enrôlement: {errors}")

    return {
        "message": f"Enrôlement réussi. {results_count} points créés pour le nuage de {name}",
        "errors": errors
    }



@router.post("/get-embedding")
async def get_embedding_only(file: UploadFile = File(...)):
    temp_path = f"audit_{uuid.uuid4()}.jpg"
    try:
        # 1. Sauvegarde l'image
        content = await file.read()
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)

        # 2. On demande le vecteur à DeepFace (On utilise Facenet comme en prod)
        res = await run_in_threadpool(dp_service.process_image, temp_path, enforce=True)
        
        if not res:
            return {"error": "Visage non détecté"}

        # 3. On renvoie le vecteur complet
        return {
            "model": dp_service.model,
            "detector": dp_service.detector,
            "confidence": res["confidence"],
            "embedding": res["embedding"] # Les 128 chiffres
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/identify", response_model=SearchResult)
async def identify(file: UploadFile = File(...)):
    temp_path = f"search_{uuid.uuid4()}.jpg"
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
            await out_file.write(await file.read())

        # Embedding du visage inconnu
        target_vec = await run_in_threadpool(dp_service.process_image, temp_path)
        target_emb = target_vec["embedding"]
        # Recherche vectorielle
        faces = db.client.collections.get("Face_Facenet")
        response = faces.query.near_vector(near_vector=target_emb, limit=1)

        if not response.objects:
            raise HTTPException(status_code=404, detail="Aucun visage reconnu")
        
        match = response.objects[0]
        return {
            "person_name": match.properties["person_name"],
            "distance": 0.0 # Weaviate peut aussi renvoyer la distance
        }
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)



from app.schemas.response import VerifyResponse # N'oublie pas l'import

@router.post("/verify", response_model=VerifyResponse)
async def verify(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # On crée deux noms de fichiers uniques
    id = uuid.uuid4()
    path1 = f"v1_{id}.jpg"
    path2 = f"v2_{id}.jpg"
    
    try:
        # Sauvegarde asynchrone des deux images
        async with aiofiles.open(path1, 'wb') as f1, aiofiles.open(path2, 'wb') as f2:
            await f1.write(await file1.read())
            await f2.write(await file2.read())

        # Comparaison (CPU Bound -> Threadpool)
        # On passe les deux chemins à la fonction de service
        result = await run_in_threadpool(dp_service.verify_faces, path1, path2)
        
        return {
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "model": result["model"],
            "detector_backend": result["detector_backend"]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    finally:
        # Nettoyage des deux fichiers
        for p in [path1, path2]:
            if os.path.exists(p):
                os.remove(p)



@router.post("/analyze", response_model=FaceAnalysis)
async def analyze(file: UploadFile = File(...)):
    temp_path = f"analyze_{uuid.uuid4()}.jpg"
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
            await out_file.write(await file.read())

        # Appel au service
        raw_results = await run_in_threadpool(dp_service.analyze_face, temp_path)
        
        # DeepFace renvoie une liste (un objet par visage détecté)
        # On prend le premier visage [0]
        res = raw_results[0] if isinstance(raw_results, list) else raw_results

        # On prépare la réponse pour qu'elle corresponde EXACTEMENT au schéma FaceAnalysis
        return {
            "age": int(res["age"]),
            "gender": res["dominant_gender"], # On prend 'dominant_gender' (str) et pas 'gender' (dict)
            "dominant_emotion": res["dominant_emotion"],
            "dominant_race": res["dominant_race"]
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    

# Utilitaire pour transformer le Base64 de React en fichier temporaire
async def save_base64_temp(base64_str: str):
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    # Retire le header "data:image/jpeg;base64," si présent
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_data = base64.b64decode(base64_str)
    async with aiofiles.open(temp_path, "wb") as f:
        await f.write(img_data)
    return temp_path

@router.post("/identify-base64")
async def identify_base64(data: Base64Request):
    temp_path = await save_base64_temp(data.img_base64)
    try:
        
        # On récupère l'objet durci
        face_result = await run_in_threadpool(dp_service.process_image, temp_path)
        
        # --- PREMIER FILTRE : Confiance de détection ---
        # Si DeepFace est sûr à moins de 90%, on rejette direct
        if face_result["confidence"] < 0.90:
            return {
                "match": False, 
                "person_name": "Image trop floue ou visage douteux",
                "confidence": face_result["confidence"]
            }

        embedding = face_result["embedding"]
        
        # --- DEUXIÈME FILTRE : Comparaison Weaviate ---
        faces = db.client.collections.get("Face_Facenet")
        response = faces.query.near_vector(
            near_vector=embedding, 
            limit=1,
            return_metadata=["distance"]
        )
        
        if not response.objects:
            return {"match": False, "person_name": "Inconnu"}

        match = response.objects[0]
        dist = match.metadata.distance
        
        # Seuil de rigeur pour Facenet : 0.40
        if dist > 0.40:
             return {"match": False, "person_name": "Non reconnu", "distance": dist}

        return {"match": True, "person_name": match.properties["person_name"], "distance": dist}

    except Exception as e:
        # Arrive si enforce_detection=True ne trouve RIEN (ex: ta main cache tout)
        return {"match": False, "person_name": "Aucun visage détecté", "error": str(e)}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)



@router.post("/analyze-base64", response_model=FaceAnalysis)
async def analyze_base64(data: Base64Request):
    temp_path = await save_base64_temp(data.img_base64)
    try:
        raw_res = await run_in_threadpool(dp_service.analyze_face, temp_path)
        res = raw_res[0] if isinstance(raw_res, list) else raw_res
        return {
            "age": int(res["age"]),
            "gender": res["dominant_gender"],
            "dominant_emotion": res["dominant_emotion"],
            "dominant_race": res["dominant_race"]
        }
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)



@router.get("/people")
async def get_registered_people():
    try:
        faces = db.client.collections.get("Face_Facenet")
        response = faces.query.fetch_objects(limit=100)
        
        # 1. On récupère tous les noms
        raw_names = [obj.properties["person_name"] for obj in response.objects]
        
        # 2. On crée une liste de noms UNIQUES
        unique_people = list(set(raw_names))
        
        # 3. On renvoie le compte des noms uniques
        return {
            "total_objects_in_db": len(raw_names), # Optionnel : pour savoir combien de photos tu as
            "count": len(unique_people),           # Le vrai nombre de personnes différentes
            "people": unique_people
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        


@router.get("/debug-db")
async def debug_db():
    faces = db.client.collections.get("Face_Facenet")
    # On récupère tout, même les vecteurs
    response = faces.query.fetch_objects(limit=100, include_vector=True)
    
    debug_list = []
    for obj in response.objects:
        debug_list.append({
            "id": str(obj.uuid),
            "name": obj.properties["person_name"],
            # On affiche juste les 3 premiers chiffres du vecteur pour comparer
            "vector_start": obj.vector["default"][:3] if obj.vector else None 
        })
    return debug_list

@router.get("/find-the-ghost")
async def find_ghost():
    try:
        faces = db.client.collections.get("Face_Facenet")
        
        # On demande explicitement d'inclure les vecteurs
        response = faces.query.fetch_objects(
            limit=100, 
            include_vector=True
        )
        
        # Le début du vecteur de l'image s1.jpg que vous avez audité
        target_start = -0.10152630507946014
        
        for obj in response.objects:
            # Sécurité : vérifier si le vecteur existe
            if obj.vector is None or "default" not in obj.vector:
                continue
            
            # Récupération du vecteur (liste de nombres)
            vec = obj.vector["default"]
            
            # Comparaison du premier chiffre avec une petite marge d'erreur (float)
            if abs(vec[0] - target_start) < 0.0001:
                return {
                    "status": "FANTÔME TROUVÉ !",
                    "db_id": str(obj.uuid),
                    "name_assigned_in_db": obj.properties.get("person_name"),
                    "full_vector_start": vec[:5],
                    "message": "Cet objet porte le nom de Belco mais possède votre signature vectorielle."
                }
                
        return {"message": "Aucun objet correspondant dans les 100 derniers. Essayez de vider la base."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/sniper-debug")
async def sniper_debug():
    # ATTENTION : Remplacez par le nom EXACT que vous utilisez dans /register
    # Est-ce "Face_Facenet" ou "Face_Collection" ? 
    # D'après vos logs, vos données semblent être dans "Face_Facenet"
    collection_name = "Face_Facenet" 
    
    try:
        faces = db.client.collections.get(collection_name)
        
        # Le vecteur exact que vous avez extrait de s1.jpg
        target_vector = [
            -0.10152630507946014, -0.2866019904613495, -0.8500612378120422, 
            -0.6363197565078735, 0.44371068477630615 # ... etc (mettez les 5 premiers)
        ]
        
        # On fait une recherche vectorielle sur ce vecteur précis
        response = faces.query.near_vector(
            near_vector=target_vector,
            limit=1,
            return_metadata=["distance"]
        )
        
        if response.objects:
            obj = response.objects[0]
            return {
                "verdict": "Coupable identifié !",
                "nom_dans_la_base": obj.properties["person_name"],
                "distance_reelle": obj.metadata.distance,
                "collection_interrogée": collection_name,
                "id_objet": str(obj.uuid)
            }
        return {"message": f"La collection {collection_name} est vide ou n'existe pas."}
    except Exception as e:
        return {"error": str(e)}