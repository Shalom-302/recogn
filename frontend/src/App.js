import React, { useRef, useState, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import './App.css';

const API_URL = "https://api-recognition.kortexai.dev/api";

// Configuration des étapes FaceID
const ENROLLMENT_STEPS = [
  { id: 0, label: "Fixez l'écran (Face)", pose: "FRONT" },
  { id: 1, label: "Approchez-vous doucement", pose: "CLOSE" },
  { id: 2, label: "Éloignez-vous un peu", pose: "FAR" },
  { id: 3, label: "Tournez la tête à gauche", pose: "LEFT" },
  { id: 4, label: "Tournez la tête à droite", pose: "RIGHT" }
];

function App() {
  const webcamRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [name, setName] = useState(""); 
  const [people, setPeople] = useState([]);

  // États pour l'enrôlement multi-poses
  const [isEnrollmentMode, setIsEnrollmentMode] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [capturedBlobs, setCapturedBlobs] = useState([]);

  useEffect(() => { listPeople(); }, []);

  const getFrame = useCallback(() => {
    return webcamRef.current ? webcamRef.current.getScreenshot() : null;
  }, [webcamRef]);

  const listPeople = async () => {
    try {
      const response = await fetch(`${API_URL}/people`);
      const data = await response.json();
      setPeople(data.people || []);
    } catch (e) { console.error("Erreur liste"); }
  };

  // --- LOGIQUE IDENTIFICATION ---
  const identify = async () => {
    const image = getFrame();
    if (!image) return;
    setLoading(true); setResult(null);
    try {
      const response = await fetch(`${API_URL}/identify-base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ img_base64: image }),
      });
      const data = await response.json();
      setResult(data.match ? `✅ ${data.person_name} (${data.distance?.toFixed(2)})` : `❌ ${data.person_name}`);
    } catch (e) { setResult("⚠️ Erreur serveur"); }
    setLoading(false);
  };

  // --- LOGIQUE ENRÔLEMENT FACEID (MULTI-POSES) ---
  const startEnrollment = () => {
    if (!name) return alert("Veuillez saisir un nom avant de commencer.");
    setIsEnrollmentMode(true);
    setCurrentStep(0);
    setCapturedBlobs([]);
    setResult(null);
  };

  const capturePose = async () => {
    const image = getFrame();
    if (!image) return;

    // Conversion immédiate en Blob pour stockage mémoire
    const blob = await (await fetch(image)).blob();
    const newBlobs = [...capturedBlobs, blob];
    setCapturedBlobs(newBlobs);

    if (currentStep < ENROLLMENT_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // Dernière étape atteinte -> Envoi au serveur
      finishEnrollment(newBlobs);
    }
  };

  const finishEnrollment = async (allBlobs) => {
    setLoading(true);
    setIsEnrollmentMode(false);
    
    const formData = new FormData();
    formData.append("name", name);
    allBlobs.forEach((blob, index) => {
      formData.append("files", blob, `pose_${index}.jpg`);
    });

    try {
      const response = await fetch(`${API_URL}/register-multi`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(`🚀 Nuage créé : ${data.message}`);
      setName("");
      listPeople();
    } catch (e) {
      setResult("⚠️ Échec de l'enrôlement multi-poses");
    }
    setLoading(false);
    setCurrentStep(0);
  };

  const analyze = async () => {
    const image = getFrame();
    if (!image) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/analyze-base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ img_base64: image }),
      });
      const data = await response.json();
      setResult(`📊 ${data.age} ans | ${data.gender} | ${data.dominant_emotion}`);
    } catch (e) { setResult("⚠️ Erreur analyse"); }
    setLoading(false);
  };

  return (
    <div className="app-container">
      <header>
        <h1>DeepFace <span className="blue">Cloud-ID</span></h1>
        <div className="status-bar">{loading ? "⚡ Traitement IA..." : "🟢 Prêt"}</div>
      </header>

      <main>
        <div className="webcam-wrapper">
          {/* Cercle FaceID dynamique */}
          {isEnrollmentMode && <div className="faceid-overlay animate-pulse"></div>}
          
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className={`webcam-view ${isEnrollmentMode ? 'enrolling' : ''}`}
            videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
          />
          
          {result && <div className="result-overlay">{result}</div>}
          
          {isEnrollmentMode && (
            <div className="instruction-box">
              <h3>{ENROLLMENT_STEPS[currentStep].label}</h3>
              <div className="progress-bar">
                {ENROLLMENT_STEPS.map(s => (
                  <div key={s.id} className={`step-dot ${currentStep >= s.id ? 'active' : ''}`}></div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="controls">
          {!isEnrollmentMode ? (
            <>
              <section className="action-group">
                <h3>🔍 Identification</h3>
                <button className="primary-btn" onClick={identify} disabled={loading}>Scanner mon visage</button>
                <button className="secondary-btn" onClick={analyze} disabled={loading}>Analyse Bio</button>
              </section>

              <section className="action-group">
                <h3>⚙️ Enrôlement Dataset</h3>
                <input 
                  type="text" 
                  placeholder="Nom complet" 
                  value={name} 
                  onChange={(e) => setName(e.target.value)}
                />
                <button className="btn-register" onClick={startEnrollment} disabled={loading}>
                  Démarrer FaceID
                </button>
                
                <div className="people-list">
                  <h4>👥 Nuages actifs ({people.length})</h4>
                  <ul>
                    {people.map((p, i) => <li key={i}>● {p}</li>)}
                  </ul>
                </div>
              </section>
            </>
          ) : (
            <section className="action-group enrollment-active">
              <h3>Enrôlement de {name}</h3>
              <p>Capturez les 5 positions requises pour le nuage de vecteurs.</p>
              <button className="capture-btn" onClick={capturePose}>
                CAPTURER LA POSE ({currentStep + 1}/5)
              </button>
              <button className="cancel-btn" onClick={() => setIsEnrollmentMode(false)}>Annuler</button>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;