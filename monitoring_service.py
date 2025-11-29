"""
Сервис мониторинга для анализа дрейфа данных
Файл: monitoring_service.py
"""
import os
import logging
from typing import List
from datetime import datetime
import json

import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

logger = logging.getLogger(__name__)

class MonitoringService:
    def __init__(self):
        self.current_data = []
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self._load_reference_data()
    
    def _load_reference_data(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        self.reference_df = pd.DataFrame(iris.data, columns=self.feature_names)
        self.reference_df['target'] = iris.target
    
    def log_prediction(self, features: List[float], prediction: int, 
                      probabilities: List[float], latency: float):
        record = {
            'timestamp': datetime.now().isoformat(),
            **dict(zip(self.feature_names, features)),
            'prediction': prediction,
            'max_probability': max(probabilities),
            'latency_ms': latency
        }
        self.current_data.append(record)
        
        if len(self.current_data) > 10000:
            self.current_data = self.current_data[-5000:]
    
    def generate_drift_report(self) -> str:
        if len(self.current_data) < 10:
            logger.warning("Недостаточно данных для анализа дрейфа")
            raise ValueError("Недостаточно данных для анализа дрейфа. Необходимо минимум 10 предсказаний.")
        
        # Создание директории для отчетов
        os.makedirs("reports", exist_ok=True)
        
        current_df = pd.DataFrame(self.current_data)
        current_features_df = current_df[self.feature_names]
        
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = self.feature_names
        
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(
            reference_data=self.reference_df[self.feature_names],
            current_data=current_features_df,
            column_mapping=column_mapping
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"drift_report_{timestamp}.html"
        report_path = os.path.join("reports", report_filename)
        
        # Сохранение отчета
        report.save_html(report_path)
        
        logger.info(f"Отчет о дрейфе сохранен: {report_path}")
        return report_path

if __name__ == "__main__":
    service = MonitoringService()
    
    # Симуляция данных
    for _ in range(100):
        features = np.random.normal([5.8, 3.0, 4.3, 1.3], [0.5, 0.4, 0.8, 0.3]).tolist()
        service.log_prediction(features, 1, [0.1, 0.8, 0.1], 50.0)
    
    service.generate_drift_report()
    print("Отчет о дрейфе сгенерирован!")
