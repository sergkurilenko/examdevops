"""
ML-сервис для обучения и инференса модели
Файл: ml_service.py
"""
import os
import pickle
import logging
from typing import Tuple, List
from datetime import datetime

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self, model_path: str = "model.pkl"):
        self.model_path = model_path
        self.model = None
        self.model_version = "1.0.0"
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target_names = ['setosa', 'versicolor', 'virginica']
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        iris = load_iris()
        return iris.data, iris.target
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42):
        logger.info("Начало обучения модели...")
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Обучение завершено. Accuracy: {accuracy:.4f}")
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_model(self):
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        model_data = {
            'model': self.model,
            'version': self.model_version,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Модель сохранена в {self.model_path}")
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_version = model_data.get('version', '1.0.0')
        logger.info(f"Модель загружена, версия: {self.model_version}")
    
    def predict(self, features: List[float]) -> Tuple[int, np.ndarray]:
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        return prediction, probabilities

if __name__ == "__main__":
    service = MLService()
    service.train_model()
    service.save_model()
    print("Модель обучена и сохранена!")
