from celery import Celery
from celery.schedules import crontab
import logging
from apps.api.settings import api_settings

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание Celery приложения
app = Celery(
    'ctg_worker',
    broker=api_settings.redis_url,
    backend=api_settings.redis_url
)

# Конфигурация Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Периодические задачи
    beat_schedule={
        'cleanup-old-sessions': {
            'task': 'apps.worker.tasks.cleanup_old_sessions',
            'schedule': crontab(minute=0),  # Каждый час
        },
        'generate-daily-reports': {
            'task': 'apps.worker.tasks.generate_daily_reports',
            'schedule': crontab(hour=1, minute=0),  # Каждый день в 1:00 UTC
        },
        'check-model-drift': {
            'task': 'apps.worker.tasks.check_model_drift',
            'schedule': crontab(hour=3, minute=0, day_of_week=1),  # Каждый понедельник в 3:00 UTC
        },
    },
)

