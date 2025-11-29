"""
Основное приложение FastAPI для ML-сервиса
Файл: main.py
"""
import time
import logging
from typing import List
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from ml_service import MLService
from monitoring_service import MonitoringService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Service API",
    description="Воспроизводимый ML-сервис с веб-интерфейсом",
    version="1.0.0"
)

# CORS middleware для работы UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_service = MLService()
monitoring_service = MonitoringService()

# Создание директории для отчетов
os.makedirs("reports", exist_ok=True)

# Монтирование статических файлов для отчетов
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_latency": 0.0,
    "start_time": datetime.now()
}

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)
    
    @validator('features')
    def validate_features(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Все признаки должны быть числами")
        if any(x < 0 for x in v):
            raise ValueError("Признаки не могут быть отрицательными")
        return v

class PredictResponse(BaseModel):
    prediction: int
    probability: List[float]
    model_version: str
    latency_ms: float

@app.on_event("startup")
async def startup_event():
    logger.info("Запуск ML-сервиса...")
    try:
        ml_service.load_model()
    except:
        logger.info("Обучение новой модели...")
        ml_service.train_model()
        ml_service.save_model()

@app.get("/")
async def root():
    """Веб-интерфейс для работы с ML-сервисом"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return {
            "message": "MLOps Service API", 
            "version": "1.0.0",
            "docs": "/docs",
            "ui": "index.html not found - please ensure it's in the same directory"
        }

@app.get("/health")
async def health_check():
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()
    return {
        "status": "healthy" if ml_service.model is not None else "unhealthy",
        "model_loaded": ml_service.model is not None,
        "uptime_seconds": uptime
    }

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    metrics["total_requests"] += 1
    start_time = time.time()
    
    try:
        prediction, probabilities = ml_service.predict(request.features)
        latency = (time.time() - start_time) * 1000
        metrics["total_latency"] += latency
        metrics["successful_predictions"] += 1
        
        monitoring_service.log_prediction(
            features=request.features,
            prediction=prediction,
            probabilities=probabilities,
            latency=latency
        )
        
        return PredictResponse(
            prediction=int(prediction),
            probability=probabilities.tolist(),
            model_version=ml_service.model_version,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        metrics["failed_predictions"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()
    avg_latency = (
        metrics["total_latency"] / metrics["successful_predictions"]
        if metrics["successful_predictions"] > 0 else 0
    )
    success_rate = (
        metrics["successful_predictions"] / metrics["total_requests"] * 100
        if metrics["total_requests"] > 0 else 0
    )
    
    return {
        "uptime_seconds": uptime,
        "total_requests": metrics["total_requests"],
        "successful_predictions": metrics["successful_predictions"],
        "failed_predictions": metrics["failed_predictions"],
        "success_rate_percent": round(success_rate, 2),
        "average_latency_ms": round(avg_latency, 2),
        "requests_per_second": round(metrics["total_requests"] / uptime, 2) if uptime > 0 else 0
    }

@app.get("/drift-report")
async def get_drift_report():
    try:
        report_path = monitoring_service.generate_drift_report()
        # Возвращаем относительный путь для веб-доступа
        web_path = f"/reports/{os.path.basename(report_path)}"
        return {
            "status": "success",
            "report_path": report_path,
            "web_path": web_path,
            "message": "Отчет о дрейфе данных сгенерирован"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
