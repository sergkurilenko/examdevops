"""
Сценарии нагрузочного тестирования
Файл: locustfile.py
"""
import random
from locust import HttpUser, task, between

class MLServiceUser(HttpUser):
    wait_time = between(0.5, 2.0)
    
    def on_start(self):
        self.client.get("/health")
    
    @task(5)
    def predict_setosa(self):
        features = [
            random.uniform(4.5, 5.5),
            random.uniform(3.0, 4.0),
            random.uniform(1.0, 2.0),
            random.uniform(0.1, 0.5)
        ]
        self.client.post("/predict", json={"features": features})
    
    @task(3)
    def predict_versicolor(self):
        features = [
            random.uniform(5.5, 6.5),
            random.uniform(2.5, 3.0),
            random.uniform(3.5, 4.5),
            random.uniform(1.0, 1.5)
        ]
        self.client.post("/predict", json={"features": features})
    
    @task(2)
    def predict_virginica(self):
        features = [
            random.uniform(6.5, 7.5),
            random.uniform(2.8, 3.5),
            random.uniform(5.0, 6.5),
            random.uniform(1.8, 2.5)
        ]
        self.client.post("/predict", json={"features": features})
    
    @task(2)
    def check_health(self):
        self.client.get("/health")
    
    @task(1)
    def check_healthcheck(self):
        self.client.get("/healthcheck")
    
    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")

# Запуск:
# locust -f locustfile.py --host=http://localhost:8000
# 
# Headless режим:
# locust -f locustfile.py --host=http://localhost:8000 \
#   --headless -u 100 -r 10 --run-time 5m --html=locust_report.html
