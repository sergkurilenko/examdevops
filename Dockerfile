FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

# Копирование файлов
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ml_service.py monitoring_service.py index.html ./

# Создание директорий
RUN mkdir -p models reports logs mlruns

# Создание непривилегированного пользователя
RUN useradd -m mlops && chown -R mlops:mlops /app
USER mlops

# Обучение модели
RUN python ml_service.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "main.py"]