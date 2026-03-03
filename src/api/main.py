"""API FastAPI para servir previsoes."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Fuel Predictor API", version="0.1.0")

class PredictionRequest(BaseModel):
    estado: str
    municipio: str
    produto: str = "GASOLINA COMUM"
    semanas_futuras: int = 4

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(req: PredictionRequest):
    # TODO: Fase 4 - integrar modelos treinados
    raise HTTPException(status_code=501, detail="Disponivel na Fase 4")
