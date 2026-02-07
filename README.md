# E-commerce Recommendation System

Магистерская диссертация: «Проектирование системы анализа поведения пользователей интернет-магазинов с использованием рекомендательных алгоритмов».

## Описание

Система анализирует поведение пользователей и предоставляет персонализированные рекомендации товаров с использованием гибридного подхода (коллаборативная + контентная фильтрация).

### Ключевые возможности

- **Гибридная рекомендательная модель** (ALS + признаки пользователей/товаров)
- **RFM-сегментация** пользователей
- **Ассоциативные правила** (Apriori/FP-Growth)
- **REST API** с кэшированием (SQLite + Redis)
- **Интерактивный UI** на Streamlit
- **MLflow** для трекинга экспериментов

## Технологический стек

| Категория | Технологии |
|-----------|------------|
| Данные | Polars, Parquet, SQLite |
| Модели | implicit (ALS), scikit-learn, mlxtend |
| API | FastAPI, uvicorn, pydantic |
| UI | Streamlit, Plotly |
| Эксперименты | MLflow, Optuna |
| Инфраструктура | Docker, docker-compose |

## Структура проекта

```
recsys/
├── src/
│   ├── data/           # Загрузка, предобработка, БД
│   ├── analysis/       # RFM, ассоциативные правила
│   ├── models/         # ALS, гибридная модель
│   ├── evaluation/     # Метрики (P@K, R@K, NDCG@K, ...)
│   └── api/            # FastAPI endpoints
├── ui/
│   ├── app.py          # Главная страница
│   └── pages/          # Analytics, Recommendations, Experiments
├── configs/            # YAML-конфигурации
├── data/
│   ├── raw/            # Исходные данные (RetailRocket)
│   ├── processed/      # train/valid/test.parquet, RFM, rules
│   └── database.sqlite # Кэш рекомендаций
├── models/             # Сохранённые модели (.pkl)
├── reports/            # CSV с результатами экспериментов
├── notebooks/          # Jupyter notebooks (EDA, ablation)
├── scripts/            # CLI-скрипты
├── mlruns/             # MLflow артефакты
└── tests/              # Pytest тесты
```

## Быстрый старт

### 1. Установка

```bash
# Клонирование
git clone <repo-url>
cd recsys

# Виртуальное окружение
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Зависимости
pip install -e .
```

### 2. Подготовка данных

```bash
# Скачать RetailRocket dataset в data/raw/

# Предобработка
python scripts/preprocess.py

# Инициализация БД
python scripts/init_database.py
```

### 3. Обучение модели

```bash
# Обучение гибридной модели
python scripts/train.py

# Оптимизация гиперпараметров (Optuna)
python scripts/optimize.py
```

### 4. Запуск

```bash
# API (http://localhost:8000)
uvicorn src.api.main:app --port 8000

# UI (http://localhost:8501)
streamlit run ui/app.py
```

## Docker

### Простой запуск

```bash
# Собрать и запустить API + UI
docker-compose up -d

# С Redis кэшем
docker-compose --profile with-redis up -d
```

### Отдельные сервисы

```bash
# Только API
docker build -t recsys-api .
docker run -p 8000:8000 recsys-api

# Только UI
docker build -f Dockerfile.ui -t recsys-ui .
docker run -p 8501:8501 recsys-ui
```

## API Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/api/v1/health` | GET | Проверка состояния |
| `/api/v1/recommend` | POST | Персональные рекомендации |
| `/api/v1/recommend/cold` | POST | Cold-start рекомендации |
| `/api/v1/similar/{item_id}` | POST | Похожие товары |
| `/api/v1/popular` | GET | Популярные товары |
| `/api/v1/cache/stats` | GET | Статистика кэша |
| `/api/v1/cache/flush` | POST | Очистка кэша |

### Пример запроса

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 11248, "n_recommendations": 10}'
```

## UI Pages

### Home (app.py)
- Обзор датасета
- Статус системы (API, модель, БД)

### Analytics
- Воронка конверсии
- RFM-сегментация
- Тепловая карта категорий
- Ассоциативные правила

### Recommendations
- Ввод user_id или выбор из примеров
- История пользователя
- Карточки рекомендаций с объяснениями
- Похожие товары

### Experiments
- Сравнение моделей
- Ablation study
- Результаты оптимизации
- Производительность API

## Метрики качества

| Метрика | Описание |
|---------|----------|
| Precision@K | Доля релевантных в top-K |
| Recall@K | Покрытие релевантных |
| NDCG@K | Качество ранжирования |
| MAP@K | Средняя точность |
| Hit Rate@K | Хотя бы 1 релевантный |
| Coverage | Покрытие каталога |

### Результаты (test set)

| Модель | NDCG@10 | Precision@10 | Recall@10 |
|--------|---------|--------------|-----------|
| ALS Baseline | 0.0046 | 0.0028 | 0.0056 |
| Hybrid (Full) | **0.0134** | **0.0044** | **0.0089** |

## Производительность API

| Concurrency | RPS | p50 | p95 | p99 |
|-------------|-----|-----|-----|-----|
| 10 users | 134 | 67ms | 119ms | 229ms |
| 50 users | 138 | 328ms | 537ms | 630ms |
| 100 users | 142 | 704ms | 854ms | 1005ms |

**Targets**: Throughput > 100 RPS ✓

## Датасет

[RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (Kaggle):

- `events.csv` — 2.7M взаимодействий (view, addtocart, transaction)
- `item_properties_*.csv` — метаданные товаров
- `category_tree.csv` — иерархия категорий

После предобработки:
- **Train**: 70% событий
- **Valid**: 10% событий
- **Test**: 20% событий (temporal split)

## Конфигурация

Все настройки в `configs/`:

- `data.yaml` — пути к данным
- `database.yaml` — SQLite кэш
- `mlflow_config.yaml` — MLflow tracking
- `best_params.yaml` — оптимальные гиперпараметры

## Разработка

```bash
# Тесты
pytest tests/ -v

# Линтинг
ruff check src/

# Форматирование
ruff format src/

# MLflow UI
mlflow ui --port 5000
```

## Лицензия

MIT License

---

**Master's Thesis Project** | 2024
