# MLOps Проект

# Выполнил: Куриленко Сергей

### Исходный код:
- **main.py** - FastAPI приложение с 7 эндпоинтами
- **ml_service.py** - ML сервис с обучением модели
- **monitoring_service.py** - Мониторинг дрейфа (EvidentlyAI)
- **locustfile.py** - Нагрузочное тестирование

### Инфраструктура:
- **requirements.txt** - все зависимости
- **Dockerfile** - контейнеризация
- **docker-compose.yml** - оркестрация

### ADR Документация:
- **ADR-001-architecture.md** - Монолит vs Микросервисы
- **ADR-002-monitoring.md** - Evidently vs Deepchecks vs WhyLabs

---

## Быстрый старт

### Шаг 1: Установка зависимостей

```bash
# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Шаг 2: Обучить модель

```bash
python ml_service.py
```

Это создаст файл `model.pkl`

### Шаг 3: Запустить сервис

```bash
python main.py
```

Сервис доступен на:
- http://localhost:8000
- Документация API: http://localhost:8000/docs

### Шаг 4: Тестирование API

```bash
# Health check
curl http://localhost:8000/health

# Предсказание
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Шаг 5: Нагрузочное тестирование

В новом терминале:

```bash
# Веб-интерфейс
locust -f locustfile.py --host=http://localhost:8000
# Откройте http://localhost:8089

# Или headless режим
locust -f locustfile.py \
  --host=http://localhost:8000 \
  --headless \
  -u 100 \
  -r 10 \
  --run-time 5m \
  --html=locust_report.html
```

### Шаг 6: Генерация отчета о дрейфе

```bash
# Через API
curl http://localhost:8000/drift-report

# Или напрямую
python monitoring_service.py
```

---

## Через Docker

```bash
# Сборка и запуск
docker-compose up -d

# Проверка
curl http://localhost:8000/health

# Логи
docker-compose logs -f

# Остановка
docker-compose down
```

---

## Структура API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Корневая страница |
| GET | `/health` | Проверка здоровья |
| GET | `/healthcheck` | Альтернативная проверка |
| POST | `/predict` | Предсказание |
| GET | `/metrics` | Метрики сервиса |
| GET | `/drift-report` | Генерация отчета |

---
