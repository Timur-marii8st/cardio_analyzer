---

[English](#english) | [Русский](#русский)

<a name="english"></a>
## Fetal Health Monitoring System (CTG Analyzer)

An advanced system for the real-time analysis of fetal cardiotocography (CTG). The application visualizes CTG data, utilizes a machine learning model to predict the risk of hypoxia, and automatically detects abnormal events like decelerations.

### Key Features

*   **Real-time Visualization**: Displays a chart with Fetal Heart Rate (FHR), Baseline, and Uterine Contractions (UA).
*   **ML-Powered Risk Prediction**: Assesses the risk of hypoxia using an ML model (LightGBM) and displays it on an intuitive gauge.
*   **Automatic Event Detection**: Identifies and lists decelerations (FHR slowdowns) on a timeline.
*   **Separate Data Upload**: Supports uploading FHR (BPM) and contraction (Uterus) data from separate CSV files.
*   **Session Management**: Create new analysis sessions and switch between existing ones, ensuring data isolation.
*   **User Authentication**: Secure login system to protect data.
*   **Background Tasks**: Uses Celery for periodic tasks like cleaning up old sessions and generating reports.

### System Architecture

The project is built on a modern microservice architecture and includes the following components:

*   **Backend**:
    *   **Framework**: FastAPI (Python)
    *   **Database**: PostgreSQL with the TimescaleDB extension for efficient time-series data handling.
    *   **Cache & Message Broker**: Redis for storing temporary session data and as a broker for Celery.
    *   **ORM**: SQLAlchemy with Alembic for database schema migrations.
    *   **Core Modules**:
        *   `packages/ctg_core`: Contains the core logic for CTG signal processing, feature extraction, and anomaly detection.
        *   `packages/ctg_ml`: Modules for training and running inference with the ML model.

*   **Frontend**:
    *   **Framework**: React with TypeScript.
    *   **Bundler**: Vite.
    *   **Visualization**: ECharts library for rendering charts and gauges.
    *   **Communication**: REST API for data uploads and WebSockets for real-time updates.

*   **Infrastructure & Deployment**:
    *   **Containerization**: Docker and Docker Compose for easy local deployment of the entire stack (API, UI, Worker, DB, Redis).
    *   **Web Server**: Nginx to serve the frontend application and proxy API requests.

### Installation and Setup

#### Option 1: Using Docker Compose (Recommended)

This is the easiest way to get all services up and running.

**Prerequisites**:
*   Docker
*   Docker Compose

**Steps**:
1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd cardio_analyzer
    ```
2.  **Configure environment variables**:
    Copy the `.env.example` file to `.env`. **You must** change the `SECRET_KEY` value to a unique, complex string.
    ```bash
    cp .env.example .env
    # Open .env and set a new SECRET_KEY
    ```3.  **Place the ML model**:
    Put your ML model file (e.g., `patient_risk_pipeline.pkl`) into the `packages/ctg_ml/` directory. The path in your `.env` file should be `ARTIFACTS_PATH=packages/ctg_ml/patient_risk_pipeline.pkl`.

4.  **Start all services**:
    ```bash
    docker-compose up -d --build
    ```
5.  **Apply database migrations**:
    Wait for the `db` container to be healthy, then run:
    ```bash
    docker-compose exec api alembic upgrade head
    ```
6.  **Create an admin user**:
    ```bash
    docker-compose exec api python -m tools.create_admin
    ```
    This will create a user with the default credentials: `admin@example.com` / `cardio_anal`.

7.  **Open the application**:
    Navigate to `http://localhost` in your browser.

#### Option 2: Local Setup without Docker

**Prerequisites**:
*   Python 3.11+
*   Node.js 20+
*   A running instance of PostgreSQL and Redis.

**Steps**:
1.  **Setup the Backend**:
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # for Linux/macOS
    # venv\Scripts\activate  # for Windows

    # Install dependencies
    pip install -r requirements.txt
    pip install -r requirements-worker.txt
    ```
2.  **Setup the Frontend**:
    ```bash
    cd apps/ui
    npm install
    cd ../..
    ```
3.  **Configure Environment**:
    Copy `.env.example` to `.env` and provide the correct `DATABASE_URL` and `REDIS_URL` for your local servers.

4.  **Setup Database and Admin User**:
    ```bash
    # Apply migrations
    alembic upgrade head

    # Create an admin user
    python -m tools.create_admin
    ```
5.  **Run Services (in separate terminals)**:
    *   **API Server**:
        ```bash
        uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
        ```
    *   **Frontend Dev Server**:
        ```bash
        cd apps/ui
        npm run dev
        ```
    *   **Celery Worker (Optional)**:
        ```bash
        celery -A apps.worker.worker worker --loglevel=info
        ```
6.  **Open the Application**:
    Navigate to `http://localhost:5173` in your browser.


### How to Use the Application

1.  **Login**: Open the web interface and log in with the credentials created during setup.
2.  **Create a New Session**: Click the **"New Session"** button. This will generate a unique ID for your analysis and clear any previous results.
3.  **Select Files**: Choose one or more CSV files for the **BPM (FHR)** channel and/or the **Uterus (contractions)** channel.
4.  **Upload and Analyze**: Click the **"Upload Files"** button. The system will upload and process the data.
5.  **View Results**: The screen will display:
    *   The main CTG chart.
    *   A gauge showing the hypoxia risk assessment.
    *   A list of detected decelerations with their characteristics.
6.  **Switching Between Sessions**: If you have multiple sessions, you can select the desired one from the dropdown menu to view its results.

#### CSV File Format

Each uploaded CSV file must contain at least two columns:
*   `time_sec` or `ts`: A timestamp (either seconds from the start of the recording or an ISO-formatted date/time).
*   `value`: The numeric measurement.

**Example (BPM):**
```csv
time_sec,value
0.0,140.5
0.25,141.2
0.5,142.0
```

### Troubleshooting

*   **"Model not loaded" error**: Ensure the `patient_risk_pipeline.pkl` file exists and that the `ARTIFACTS_PATH` in your `.env` file is correct.
*   **WebSocket fails to connect**: Check that the API server is running on port 8000 and that the `CORS_ORIGINS` in your `.env` file includes the correct frontend address.
*   **Charts do not appear after upload**: Check the backend terminal for errors. The uploaded files might not contain enough data for analysis (a minimum of 5 minutes of recording is required).

---

<a name="русский"></a>
## Система мониторинга состояния плода (CTG Analyzer)

Продвинутая система для анализа кардиотокографии (КТГ) плода в реальном времени. Приложение визуализирует данные КТГ, использует модель машинного обучения для прогнозирования риска гипоксии и автоматически обнаруживает аномальные события, такие как децелерации.

### Основные возможности

*   **Визуализация в реальном времени**: Отображение графика с частотой сердечных сокращений плода (FHR), базовой линией (Baseline) и сокращениями матки (UA).
*   **ML-прогнозирование риска**: Оценка риска гипоксии с помощью ML-модели (LightGBM) и его отображение на интуитивно понятном датчике.
*   **Автоматическое обнаружение событий**: Идентификация и отображение децелераций (замедлений ЧСС) на временной шкале.
*   **Раздельная загрузка данных**: Поддержка загрузки данных ЧСС (BPM) и сокращений (Uterus) из отдельных CSV-файлов.
*   **Управление сессиями**: Возможность создавать новые сессии для анализа и переключаться между существующими, обеспечивая изоляцию данных.
*   **Аутентификация пользователей**: Безопасный вход в систему для защиты данных.
*   **Фоновые задачи**: Использование Celery для периодических задач, таких как очистка старых сессий и генерация отчетов.

### Архитектура системы

Проект построен на современной микросервисной архитектуре и включает в себя следующие компоненты:

*   **Бэкенд (Backend)**:
    *   **Фреймворк**: FastAPI (Python)
    *   **База данных**: PostgreSQL с расширением TimescaleDB для эффективной работы с временными рядами.
    *   **Кэш и брокер сообщений**: Redis для хранения временных данных сессий и для Celery.
    *   **ORM**: SQLAlchemy с Alembic для миграций схемы БД.
    *   **Ключевые модули**:
        *   `packages/ctg_core`: Основная логика обработки сигналов КТГ, извлечения признаков и обнаружения аномалий.
        *   `packages/ctg_ml`: Модули для обучения и инференса ML-модели.

*   **Фронтенд (Frontend)**:
    *   **Фреймворк**: React с TypeScript.
    *   **Сборщик**: Vite.
    *   **Визуализация**: Библиотека ECharts для построения графиков и датчиков.
    *   **Коммуникация**: REST API для загрузки данных и WebSocket для получения обновлений в реальном времени.

*   **Инфраструктура и развертывание**:
    *   **Контейнеризация**: Docker и Docker Compose для простого локального развертывания всего стека (API, UI, Worker, DB, Redis).
    *   **Веб-сервер**: Nginx для обслуживания фронтенда и проксирования запросов к API.

### Установка и запуск

#### Вариант 1: С использованием Docker Compose (Рекомендуемый)

Это самый простой способ запустить все сервисы.

**Предварительные требования**:
*   Docker
*   Docker Compose

**Шаги**:
1.  **Клонируйте репозиторий**:
    ```bash
    git clone <your-repository-url>
    cd cardio_analyzer
    ```
2.  **Настройте переменные окружения**:
    Скопируйте файл `.env.example` в `.env`. **Обязательно** измените значение `SECRET_KEY` на уникальную и сложную строку.
    ```bash
    cp .env.example .env
    # Откройте .env и установите новый SECRET_KEY
    ```
3.  **Разместите ML-модель**:
    Поместите ваш файл с ML-моделью (например, `patient_risk_pipeline.pkl`) в директорию `packages/ctg_ml/`. Путь к файлу в `.env` должен быть `ARTIFACTS_PATH=packages/ctg_ml/patient_risk_pipeline.pkl`.

4.  **Запустите все сервисы**:
    ```bash
    docker-compose up -d --build
    ```
5.  **Примените миграции базы данных**:
    Дождитесь, пока контейнер `db` станет здоровым, а затем выполните:
    ```bash
    docker-compose exec api alembic upgrade head
    ```
6.  **Создайте администратора**:
    ```bash
    docker-compose exec api python -m tools.create_admin
    ```
    По умолчанию будут созданы учетные данные: `admin@example.com` / `cardio_anal`.

7.  **Откройте приложение**:
    Перейдите в браузере по адресу `http://localhost`.

#### Вариант 2: Локальный запуск без Docker

**Предварительные требования**:
*   Python 3.11+
*   Node.js 20+
*   Запущенные PostgreSQL и Redis.

**Шаги**:
1.  **Настройка бэкенда**:
    ```bash
    # Создайте и активируйте виртуальное окружение
    python -m venv venv
    source venv/bin/activate  # для Linux/macOS
    # venv\Scripts\activate  # для Windows

    # Установите зависимости
    pip install -r requirements.txt
    pip install -r requirements-worker.txt
    ```
2.  **Настройка фронтенда**:
    ```bash
    cd apps/ui
    npm install
    cd ../..
    ```
3.  **Настройка окружения**:
    Скопируйте `.env.example` в `.env` и укажите корректные `DATABASE_URL` и `REDIS_URL` для ваших локальных серверов.

4.  **Настройка базы данных и администратора**:
    ```bash
    # Примените миграции
    alembic upgrade head

    # Создайте администратора
    python -m tools.create_admin
    ```
5.  **Запуск сервисов (в разных терминалах)**:
    *   **API-сервер**:
        ```bash
        uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
        ```
    *   **Фронтенд**:
        ```bash
        cd apps/ui
        npm run dev
        ```
    *   **Celery Worker (опционально)**:
        ```bash
        celery -A apps.worker.worker worker --loglevel=info
        ```
6.  **Откройте приложение**:
    Перейдите в браузере по адресу `http://localhost:5173`.


### Использование приложения

1.  **Вход в систему**: Откройте веб-интерфейс и войдите, используя учетные данные, созданные на этапе установки.
2.  **Создание новой сессии**: Нажмите кнопку **"New Session"**. Это создаст уникальный идентификатор для вашего анализа и очистит предыдущие результаты.
3.  **Выбор файлов**: Выберите один или несколько CSV-файлов для канала **BPM (ЧСС)** и/или канала **Uterus (сокращения)**.
4.  **Загрузка и анализ**: Нажмите кнопку **"Upload Files"**. Система загрузит и обработает данные.
5.  **Просмотр результатов**: На экране отобразятся:
    *   Основной график КТГ.
    *   Датчик с оценкой риска гипоксии.
    *   Список обнаруженных децелераций с их характеристиками.
6.  **Переключение между сессиями**: Если у вас несколько сессий, вы можете выбрать нужную из выпадающего списка, чтобы просмотреть ее результаты.

#### Формат CSV-файлов

Каждый загружаемый CSV-файл должен содержать как минимум два столбца:
*   `time_sec` или `ts`: Временная метка (секунды от начала записи или дата/время в формате ISO).
*   `value`: Числовое значение измерения.

**Пример (BPM):**
```csv
time_sec,value
0.0,140.5
0.25,141.2
0.5,142.0
```

### Устранение неполадок

*   **Ошибка "Модель не загружена"**: Убедитесь, что файл `patient_risk_pipeline.pkl` существует и путь к нему в `.env` (`ARTIFACTS_PATH`) указан верно.
*   **WebSocket не подключается**: Проверьте, что API-сервер запущен на порту 8000 и что в `.env` (`CORS_ORIGINS`) указан правильный адрес фронтенда.
*   **Графики не отображаются после загрузки**: Проверьте терминал бэкенда на наличие ошибок. Возможно, загруженные файлы содержат недостаточно данных для анализа (требуется минимум 5 минут записи).