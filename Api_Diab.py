from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import sys
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Diabetes Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9090"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


model = None
scaler = None
features = None


@app.on_event("startup")
async def load_artefacts():
    global model, scaler, features
    try:
        logger.info("Début du chargement des artefacts...")
        
        model = joblib.load('diabetes_model.pkl')
        logger.info("Modèle chargé avec succès")
        
        scaler = joblib.load('scaler.pkl')
        logger.info("Scaler chargé avec succès")
        
        features = joblib.load('features.pkl')
        logger.info(f"Features chargées : {features}")
        
    except FileNotFoundError as e:
        logger.critical(f"Fichier manquant: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Erreur au chargement: {str(e)}")
        sys.exit(1)

class DiabetesRequest(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class PredictionResult(BaseModel):
    prediction: int
    probability: float
    interpretation: str
    confidence: str

@app.post("/predict", response_model=PredictionResult)
async def predict_diabetes(input_data: DiabetesRequest):
    try:
        logger.info(f"Requête reçue: {input_data.dict()}")
        
      
        if list(input_data.dict().keys()) != features:
            error_msg = f"Mismatch features: Reçu {list(input_data.dict().keys())} vs Attendus {features}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
      
        input_df = pd.DataFrame([input_data.dict()], columns=features)
 
        scaled_data = scaler.transform(input_df)
        

        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1]
        
        logger.info(f"Prédiction réussie - Résultat: {prediction[0]}")
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "interpretation": "Diabétique" if prediction[0] == 1 else "Non diabétique",
            "confidence": f"{probability*100:.1f}%"
        }
        
    except Exception as e:
        logger.error(f"Erreur de prédiction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur lors du traitement"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_loaded": features is not None,
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )