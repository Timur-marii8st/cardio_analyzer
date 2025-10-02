from celery import shared_task
from datetime import datetime, timedelta
import pandas as pd
import logging
import json

from packages.ctg_core.session_manager import RedisSessionManager
from apps.api.services.storage import Storage, DbConfig
from apps.api.settings import api_settings

logger = logging.getLogger(__name__)

@shared_task(name='apps.worker.tasks.cleanup_old_sessions')
def cleanup_old_sessions():
    """Очистка старых неактивных сессий"""
    try:
        session_manager = RedisSessionManager(redis_url=api_settings.redis_url)
        storage = Storage(DbConfig(url=api_settings.database_url))
        
        # Получаем все сессии
        sessions = session_manager.get_all_sessions()
        cleaned_count = 0
        
        for session_id in sessions:
            session = session_manager.get_session(session_id)
            if session and session.updated_at:
                last_update = datetime.fromisoformat(session.updated_at)
                if datetime.utcnow() - last_update > timedelta(hours=24):
                    # Сохраняем финальное состояние в БД перед удалением
                    if session.last_risk_prob is not None:
                        storage.save_risk(
                            session_id,
                            datetime.fromisoformat(session.last_risk_ts),
                            session.last_risk_prob,
                            session.last_risk_band
                        )
                    
                    # Удаляем из Redis
                    session_manager.delete_session(session_id)
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} inactive sessions")
        return {"cleaned_sessions": cleaned_count}
    
    except Exception as e:
        logger.error(f"Error in cleanup_old_sessions: {e}")
        raise

@shared_task(name='apps.worker.tasks.generate_daily_reports')
def generate_daily_reports():
    """Генерация ежедневных отчетов по всем сессиям"""
    try:
        storage = Storage(DbConfig(url=api_settings.database_url))
        
        # Получаем данные за последние 24 часа
        since = datetime.utcnow() - timedelta(days=1)
        
        with storage.engine.begin() as conn:
            # Статистика по сессиям
            sessions_stats = conn.execute(
                """
                SELECT 
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(DISTINCT patient_id) as unique_patients,
                    AVG(CASE WHEN band = 'high' THEN 1 ELSE 0 END) as high_risk_rate
                FROM sessions s
                JOIN risk_records r ON s.session_id = r.session_id
                WHERE r.ts >= %s
                """,
                (since,)
            ).fetchone()
            
            # Статистика по событиям
            events_stats = conn.execute(
                """
                SELECT 
                    type,
                    COUNT(*) as count,
                    AVG(dur_s) as avg_duration,
                    MAX(max_drop) as max_drop
                FROM events
                WHERE start_ts >= %s
                GROUP BY type
                """,
                (since,)
            ).fetchall()
        
        report = {
            "report_date": datetime.utcnow().isoformat(),
            "period_start": since.isoformat(),
            "sessions": {
                "total": sessions_stats.total_sessions,
                "unique_patients": sessions_stats.unique_patients,
                "high_risk_rate": float(sessions_stats.high_risk_rate or 0)
            },
            "events": [
                {
                    "type": row.type,
                    "count": row.count,
                    "avg_duration": float(row.avg_duration or 0),
                    "max_drop": float(row.max_drop or 0)
                }
                for row in events_stats
            ]
        }
        
        # Сохраняем отчет (можно отправить по email или сохранить в S3)
        logger.info(f"Daily report generated: {json.dumps(report)}")
        
        return report
    
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        raise

@shared_task(name='apps.worker.tasks.check_model_drift')
def check_model_drift():
    """Проверка дрифта модели на основе последних предсказаний"""
    try:
        storage = Storage(DbConfig(url=api_settings.database_url))
        
        # Получаем предсказания за последнюю неделю
        since = datetime.utcnow() - timedelta(days=7)
        
        with storage.engine.begin() as conn:
            result = conn.execute(
                """
                SELECT 
                    hypoxia_prob,
                    band,
                    ts
                FROM risk_records
                WHERE ts >= %s
                ORDER BY ts
                """,
                (since,)
            )
            
            predictions = pd.DataFrame(result.fetchall(), columns=['prob', 'band', 'ts'])
        
        if predictions.empty:
            logger.warning("No predictions found for drift check")
            return {"status": "no_data"}
        
        # Анализ распределения предсказаний
        drift_metrics = {
            "mean_prob": float(predictions['prob'].mean()),
            "std_prob": float(predictions['prob'].std()),
            "high_risk_ratio": float((predictions['band'] == 'high').mean()),
            "sample_size": len(predictions)
        }
        
        # Проверка на аномальные паттерны
        # Например, если среднее значение вероятности сильно отклонилось
        baseline_mean = 0.15  # Ожидаемое среднее из обучающих данных
        if abs(drift_metrics['mean_prob'] - baseline_mean) > 0.1:
            logger.warning(f"Potential model drift detected: mean_prob={drift_metrics['mean_prob']}")
            drift_metrics['drift_detected'] = True
        else:
            drift_metrics['drift_detected'] = False
        
        logger.info(f"Model drift check completed: {drift_metrics}")
        
        return drift_metrics
    
    except Exception as e:
        logger.error(f"Error checking model drift: {e}")
        raise

@shared_task(name='apps.worker.tasks.generate_patient_report')
def generate_patient_report(session_id: str, patient_id: str = None):
    """Генерация PDF отчета для конкретного пациента"""
    try:
        storage = Storage(DbConfig(url=api_settings.database_url))
        
        # Получаем все данные по сессии
        with storage.engine.begin() as conn:
            # Риски
            risks = pd.read_sql(
                """
                SELECT ts, hypoxia_prob, band 
                FROM risk_records 
                WHERE session_id = %s 
                ORDER BY ts
                """,
                conn,
                params=(session_id,)
            )
            
            # События
            events = pd.read_sql(
                """
                SELECT type, start_ts, end_ts, dur_s, severity 
                FROM events 
                WHERE session_id = %s 
                ORDER BY start_ts
                """,
                conn,
                params=(session_id,)
            )
        
        report = {
            "session_id": session_id,
            "patient_id": patient_id,
            "generated_at": datetime.utcnow().isoformat(),
            "risk_summary": {
                "max_prob": float(risks['hypoxia_prob'].max()) if not risks.empty else 0,
                "mean_prob": float(risks['hypoxia_prob'].mean()) if not risks.empty else 0,
                "high_risk_periods": len(risks[risks['band'] == 'high'])
            },
            "events_summary": {
                "total_events": len(events),
                "decelerations": len(events[events['type'] == 'deceleration']) if not events.empty else 0
            }
        }
        
        logger.info(f"Patient report generated for session {session_id}")
        return report
    
    except Exception as e:
        logger.error(f"Error generating patient report: {e}")
        raise