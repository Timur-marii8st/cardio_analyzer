# CTG Fetal Health Monitoring System

Система мониторинга состояния плода на основе кардиотокографии (CTG) с ML-предсказанием риска гипоксии.

## Новые возможности

### Загрузка двух отдельных файлов

Теперь система поддерживает раздельную загрузку данных:
- **BPM файл** - данные частоты сердечных сокращений плода (Heart Rate)
- **Uterus файл** - данные маточной активности (Contractions)

Можно загрузить:
- Только BPM файл - система рассчитает прогноз по доступным данным
- Только Uterus файл - система использует эти данные
- Оба файла - наиболее точный прогноз с учетом обоих каналов

### Формат CSV файлов

Каждый CSV файл должен содержать колонки:
- `ts` или `time_sec` - временная метка
- `value` - значение показателя

Пример BPM файла:
```csv
time_sec,value
0.0,140.5
0.25,141.2
0.5,142.0
```

Пример Uterus файла:
```csv
time_sec,value
0.0,10.5
0.25,12.3
0.5,15.8
```

## Установка и запуск

### 1. Установка зависимостей

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd apps/ui
npm install
```

### 2. Размещение ML модели

Поместите файл `patient_risk_pipeline.pkl` в директорию:
```
packages/ctg_ml/patient_risk_pipeline.pkl
```

Модель должна содержать:
```python
{
    "model": <trained_lgbm_model>,
    "scaler": <StandardScaler>,
    "features": <list_of_feature_names>
}
```

### 3. Настройка окружения

Скопируйте `.env.example` в `.env` и настройте:
```bash
cp .env.example .env
```

Обязательно установите:
- `DATABASE_URL` - подключение к PostgreSQL
- `REDIS_URL` - подключение к Redis
- `SECRET_KEY` - секретный ключ (минимум 32 символа)
- `ARTIFACTS_PATH` - путь к pkl файлу модели

### 4. Миграции базы данных

```bash
alembic upgrade head
```

### 5. Создание администратора

```bash
python -m tools.create_admin
```

### 6. Запуск сервисов

```bash
# Backend API
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd apps/ui
npm run dev

# Worker (опционально, для фоновых задач)
celery -A apps.worker.worker worker --loglevel=info
```

## Использование

1. Откройте браузер: `http://localhost:5173`
2. Войдите с учетными данными администратора
3. Выберите BPM файл и/или Uterus файл
4. Нажмите "Upload Files"
5. Система обработает данные и отобразит:
   - График CTG с FHR, baseline и UA
   - Риск гипоксии (gauge)
   - Обнаруженные децелерации
   - Временную линию событий

## Архитектура

### Backend (FastAPI)
- `apps/api/routers/adapters.py` - прием CSV файлов
- `apps/api/services/inference.py` - ML предсказания
- `apps/api/services/streaming.py` - real-time обработка
- `packages/ctg_core/features.py` - извлечение CTG-признаков

### Frontend (React + TypeScript)
- `apps/ui/src/App.tsx` - главный компонент с двумя кнопками загрузки
- `apps/ui/src/api/api.ts` - API клиент
- Компоненты: CtgChart, RiskGauge, EventTimeline

### ML Pipeline
- Модель: LightGBM с StandardScaler
- Входные признаки: 21 CTG-параметр
- Выход: вероятность гипоксии [0, 1]

## CTG Признаки

Система вычисляет следующие признаки:
1. **baseline value** - базовая линия ЧСС
2. **accelerations** - акцелерации (в секунду)
3. **fetal_movement** - движения плода
4. **uterine_contractions** - схватки
5. **light_decelerations** - легкие децелерации
6. **severe_decelerations** - тяжелые децелерации
7. **prolongued_decelerations** - продолжительные децелерации
8. **abnormal_short_term_variability** - аномальная STV (%)
9. **mean_value_of_short_term_variability** - средняя STV
10. **percentage_of_time_with_abnormal_long_term_variability** - аномальная LTV (%)
11. **mean_value_of_long_term_variability** - средняя LTV
12-21. **histogram_*** - гистограммные характеристики

## Troubleshooting

### Модель не загружается
Проверьте:
- Путь в `ARTIFACTS_PATH` корректен
- Файл `patient_risk_pipeline.pkl` существует
- Формат файла: `{"model": ..., "scaler": ..., "features": ...}`

### Ошибки при обработке CSV
- Убедитесь что CSV содержит колонки `time_sec` или `ts` и `value`
- Проверьте разделитель (`,`, `;`, `\t`)
- Убедитесь что значения числовые

### WebSocket не подключается
- Проверьте что backend запущен на порту 8000
- Проверьте настройки CORS в `.env`

## Docker Deployment

```bash
# Запуск всех сервисов
docker-compose up -d

# Логи
docker-compose logs -f api

# Остановка
docker-compose down
```