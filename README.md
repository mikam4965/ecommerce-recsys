# E-commerce Recommendation System

Система анализа поведения пользователей интернет-магазинов с использованием рекомендательных алгоритмов.

## Описание

Система анализирует поведение пользователей и предоставляет персонализированные рекомендации товаров с использованием 6 различных подходов: коллаборативная фильтрация, контентная фильтрация, гибридные модели, метод ближайших соседей и нейросетевые модели.

### Ключевые возможности

- **6 рекомендательных моделей**: ALS, ContentBased, ItemKNN, NCF, GRU4Rec, Hybrid
- **A/B тестирование** с статистической значимостью (Welch's t-test)
- **CTR-анализ** и оценка финансового влияния рекомендаций
- **RFM-сегментация** пользователей
- **Ассоциативные правила** (Apriori/FP-Growth)
- **REST API** с кэшированием (SQLite + Redis)
- **Интерактивный UI** на Streamlit
- **MLflow** для трекинга экспериментов
- **Docker** для контейнеризации

## Технологический стек

| Категория | Технологии |
|-----------|------------|
| Данные | Polars, Parquet, SQLite |
| Модели | implicit (ALS), PyTorch (NCF, GRU4Rec), scikit-learn, mlxtend |
| API | FastAPI, uvicorn, pydantic |
| UI | Streamlit, Plotly |
| Эксперименты | MLflow, Optuna |
| Инфраструктура | Docker, docker-compose, Redis |

## Рекомендательные модели

| Модель | Тип | Описание |
|--------|-----|----------|
| **ALS** | Коллаборативная фильтрация | Матричная факторизация (Alternating Least Squares) |
| **ContentBased** | Контентная фильтрация | Категорийные профили пользователей + cosine similarity |
| **Hybrid** | Гибридная модель | CF + признаки пользователей/товаров |
| **ItemKNN** | Метод ближайших соседей | Item-based k-NN, cosine similarity (k=50) |
| **NCF** | Нейросетевая модель | Neural Collaborative Filtering (MLP + embedding) |
| **GRU4Rec** | Нейросетевая модель | Сессионная рекомендация на основе GRU (RNN) |

## Результаты (RetailRocket, 1000 пользователей)

| Модель | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Время обучения |
|--------|-------------|-----------|---------|------------|----------------|
| **ContentBased** | **0.0044** | **0.0150** | **0.0121** | **3.6%** | 6.0с |
| ItemKNN | 0.0023 | 0.0084 | 0.0065 | 2.0% | 150.8с |
| GRU4Rec | 0.0022 | 0.0096 | 0.0050 | 1.8% | 497с |
| ALS | 0.0021 | 0.0048 | 0.0042 | 1.7% | 12.7с |
| NCF | 0.0012 | 0.0043 | 0.0038 | 1.0% | 185с |
| Hybrid | 0.0008 | 0.0020 | 0.0011 | 0.6% | 19.1с |

## Структура проекта

```
recsys/
├── src/
│   ├── data/           # Загрузка, предобработка, БД
│   ├── analysis/       # RFM, воронка, ассоциативные правила
│   ├── models/         # ALS, ContentBased, ItemKNN, NCF, GRU4Rec, Hybrid
│   ├── evaluation/     # Метрики, A/B тестирование
│   └── api/            # FastAPI endpoints
├── ui/
│   ├── app.py          # Главная страница
│   └── pages/          # Analytics, Recommendations, Experiments
├── scripts/            # Обучение, оптимизация, бенчмарки
├── configs/            # YAML-конфигурации
├── data/
│   ├── raw/            # Исходные данные (RetailRocket)
│   ├── processed/      # train/valid/test.parquet, RFM, rules
│   └── database.sqlite # Кэш рекомендаций
├── models/             # Сохранённые модели (.pkl)
├── reports/            # CSV с результатами экспериментов
├── notebooks/          # Jupyter notebooks (EDA, ablation)
├── mlruns/             # MLflow артефакты
└── tests/              # Pytest тесты
```

## Быстрый старт

### 1. Установка

```bash
git clone <repo-url>
cd recsys

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

pip install -e .
```

### 2. Подготовка данных

```bash
# Скачать RetailRocket dataset в data/raw/
python scripts/preprocess.py
python scripts/init_database.py
```

### 3. Обучение моделей

```bash
# Обучение всех 6 моделей + A/B тесты
python scripts/train_ncf_and_ab_test.py

# Оптимизация гиперпараметров (Optuna)
python scripts/optimize.py
```

### 4. Запуск

```bash
# API (http://localhost:8000)
.venv\Scripts\uvicorn src.api.main:app --port 8000

# UI (http://localhost:8501)
.venv\Scripts\streamlit run ui/app.py
```

## Docker

```bash
# API + UI
docker-compose up -d

# С Redis кэшем
docker-compose --profile with-redis up -d
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

## UI

| Страница | Описание |
|----------|----------|
| **Home** | Обзор датасета, статус системы (API, модель, БД) |
| **Analytics** | Воронка конверсии, RFM-сегментация, тепловая карта, ассоциативные правила |
| **Recommendations** | Ввод user_id, история пользователя, карточки рекомендаций |
| **Experiments** | Сравнение 6 моделей, A/B тесты, CTR, финансовый анализ, ablation study |

## Метрики качества

| Метрика | Описание |
|---------|----------|
| Precision@K | Доля релевантных в top-K |
| Recall@K | Покрытие релевантных |
| NDCG@K | Качество ранжирования |
| MAP@K | Средняя точность |
| Hit Rate@K | Хотя бы 1 релевантный в top-K |
| MRR@K | Средний обратный ранг |

## Датасет

[RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (Kaggle):

- `events.csv` — 2.7M взаимодействий (view, addtocart, transaction)
- `item_properties_*.csv` — метаданные товаров
- `category_tree.csv` — иерархия категорий

После предобработки:
- **Train**: 70% событий (~1.9M)
- **Valid**: 10% событий
- **Test**: 20% событий (temporal split)

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
