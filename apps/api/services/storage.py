from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, List, Dict
from datetime import datetime, date
import logging
import secrets
from sqlalchemy import func

from sqlalchemy import (
    create_engine, Engine, Table, Column, MetaData, String, Float, DateTime, Integer, ForeignKey, UniqueConstraint, Boolean, text, insert, Date, select, update, delete
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

@dataclass
class DbConfig:
    url: str

class Storage:
    def __init__(self, cfg: DbConfig):
        if cfg:
            self.engine: Engine = create_engine(cfg.url, pool_pre_ping=True, future=True)
            self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False, future=True)
        else:
            self.engine = None
            self.SessionLocal = None
            
        self.meta = MetaData()
        # Таблицы
        self.devices = Table(
            "devices", self.meta,
            Column("device_id", String, primary_key=True),
            Column("created_at", DateTime(timezone=True), server_default=text("NOW()"))
        )
        self.patients = Table(
            "patients", self.meta,
            Column("patient_id", String, primary_key=True, default=lambda: f"pat_{secrets.token_urlsafe(16)}"),
            Column("medical_record_number", String, unique=True, nullable=False),
            Column("full_name", String, nullable=False),
            Column("date_of_birth", Date, nullable=True),
            Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
            Column("updated_at", DateTime(timezone=True), server_default=text("NOW()"), onupdate=datetime.now)
        )
        self.sessions = Table(
            "sessions", self.meta,
            Column("session_id", String, primary_key=True),
            Column("device_id", String, ForeignKey("devices.device_id"), nullable=True),
            Column("patient_id", String, ForeignKey("patients.patient_id"), nullable=True),
            Column("status", String, default="active"),
            Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
            Column("updated_at", DateTime(timezone=True), server_default=text("NOW()"))
        )
        # Ряды: 1 запись на метку времени (bpm/ua в колонках)
        self.samples = Table(
            "ctg_samples", self.meta,
            Column("session_id", String, primary_key=True),
            Column("ts", DateTime(timezone=True), primary_key=True),
            Column("bpm", Float, nullable=True),
            Column("ua", Float, nullable=True),
        )
        self.risks = Table(
            "risk_records", self.meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String, ForeignKey("sessions.session_id")),
            Column("ts", DateTime(timezone=True), nullable=False),
            Column("hypoxia_prob", Float, nullable=False),
            Column("band", String, nullable=False),
        )
        self.events = Table(
            "events", self.meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("session_id", String, ForeignKey("sessions.session_id")),
            Column("type", String, nullable=False),  # deceleration/acceleration/contraction/...
            Column("start_ts", DateTime(timezone=True), nullable=False),
            Column("end_ts", DateTime(timezone=True), nullable=True),
            Column("dur_s", Float, nullable=True),
            Column("min_bpm", Float, nullable=True),
            Column("max_drop", Float, nullable=True),
            Column("severity", String, nullable=True),
            UniqueConstraint("session_id", "type", "start_ts", "end_ts", name="uq_event_unique")
        )

    def upsert_device(self, device_id: Optional[str]):
        if not device_id: return
        with self.engine.begin() as con:
            stmt = pg_insert(self.devices).values(device_id=device_id).on_conflict_do_nothing()
            con.execute(stmt)

    def upsert_session(self, session_id: str, device_id: Optional[str], patient_id: Optional[str]):
        with self.engine.begin() as con:
            if device_id:
                self.upsert_device(device_id)
            stmt = pg_insert(self.sessions).values(
                session_id=session_id, device_id=device_id, patient_id=patient_id, status="active"
            ).on_conflict_do_update(
                index_elements=["session_id"],
                set_={"device_id": device_id or self.sessions.c.device_id,
                      "patient_id": patient_id or self.sessions.c.patient_id,
                      "updated_at": text("NOW()")}
            )
            con.execute(stmt)

    def get_session_report_data(self, session_id: str) -> Optional[Dict]:
        """Gathers all data required for a comprehensive session report."""
        with self.engine.begin() as con:
            # 1. Get session and patient info
            session_info_stmt = select(self.sessions, self.patients).join(
                self.patients, self.sessions.c.patient_id == self.patients.c.patient_id, isouter=True
            ).where(self.sessions.c.session_id == session_id)
            session_result = con.execute(session_info_stmt).fetchone()
            if not session_result:
                return None  # Session not found

            # 2. Get time boundaries from samples
            time_bounds_stmt = select(
                func.min(self.samples.c.ts).label("min_ts"),
                func.max(self.samples.c.ts).label("max_ts")
            ).where(self.samples.c.session_id == session_id)
            time_bounds = con.execute(time_bounds_stmt).fetchone()

            total_duration_minutes = 0.0
            if time_bounds and time_bounds.min_ts and time_bounds.max_ts:
                duration_delta = time_bounds.max_ts - time_bounds.min_ts
                total_duration_minutes = duration_delta.total_seconds() / 60.0

            # 3. Get risk summary
            risk_summary_stmt = select(
                func.avg(self.risks.c.hypoxia_prob).label("mean_prob"),
                func.max(self.risks.c.hypoxia_prob).label("max_prob"),
                func.count(self.risks.c.band).filter(self.risks.c.band == 'normal').label("normal_count"),
                func.count(self.risks.c.band).filter(self.risks.c.band == 'elevated').label("elevated_count"),
                func.count(self.risks.c.band).filter(self.risks.c.band == 'high').label("high_count"),
            ).where(self.risks.c.session_id == session_id)
            risk_result = con.execute(risk_summary_stmt).fetchone()

            # Assume each risk record represents a 5-second interval from the streamer
            risk_interval_minutes = 5.0 / 60.0
            risk_summary_data = {
                "mean_hypoxia_prob": risk_result.mean_prob,
                "max_hypoxia_prob": risk_result.max_prob,
                "minutes_in_normal_band": (risk_result.normal_count or 0) * risk_interval_minutes,
                "minutes_in_elevated_band": (risk_result.elevated_count or 0) * risk_interval_minutes,
                "minutes_in_high_band": (risk_result.high_count or 0) * risk_interval_minutes,
            }

            # 4. Get event summary for decelerations
            event_summary_stmt = select(
                self.events.c.severity,
                func.count(self.events.c.id).label("count")
            ).where(
                self.events.c.session_id == session_id,
                self.events.c.type == 'deceleration'
            ).group_by(self.events.c.severity)
            event_results = con.execute(event_summary_stmt).fetchall()
            
            event_summary_data = {
                "total_decelerations": 0, "light_decelerations": 0,
                "severe_decelerations": 0, "prolonged_decelerations": 0
            }
            for row in event_results:
                if row.severity == 'light': event_summary_data["light_decelerations"] = row.count
                elif row.severity == 'severe': event_summary_data["severe_decelerations"] = row.count
                elif row.severity == 'prolonged': event_summary_data["prolonged_decelerations"] = row.count
            
            event_summary_data["total_decelerations"] = sum(event_summary_data.values())

            return {
                "session_info": session_result._asdict(),
                "time_bounds": time_bounds,
                "total_duration_minutes": total_duration_minutes,
                "risk_summary": risk_summary_data,
                "event_summary": event_summary_data
            }

    def insert_samples(self, session_id: str, samples: List[Dict]):
        # Пивотируем: для каждой ts — bpm/ua
        from collections import defaultdict
        rows = defaultdict(lambda: {"session_id": session_id, "bpm": None, "ua": None})
        for s in samples:
            ts = s["ts"]
            ch = s["channel"]; v = float(s["value"])
            r = rows[ts]; r["ts"] = ts
            if ch == "bpm": r["bpm"] = v
            elif ch == "uterus": r["ua"] = v
        values = list(rows.values())
        if not values: return
        with self.engine.begin() as con:
            stmt = pg_insert(self.samples).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["session_id","ts"],
                set_={"bpm": stmt.excluded.bpm, "ua": stmt.excluded.ua}
            )
            con.execute(stmt)

    def save_risk(self, session_id: str, ts: datetime, prob: float, band: str):
        with self.engine.begin() as con:
            con.execute(insert(self.risks).values(session_id=session_id, ts=ts, hypoxia_prob=prob, band=band))

    def save_new_decel_events(self, session_id: str, events: List[Dict]):
        # Ожидаем закрытые эпизоды (есть start_ts и end_ts)
        if not events: return
        rows = []
        for e in events:
            if not e.get("start_ts") or not e.get("end_ts"):
                continue
            rows.append(dict(
                session_id=session_id, type="deceleration",
                start_ts=e["start_ts"], end_ts=e["end_ts"],
                dur_s=e.get("dur_s"), min_bpm=e.get("min_bpm"), max_drop=e.get("max_drop"),
                severity=_severity_from_drop(e.get("max_drop", 0.0), e.get("dur_s", 0.0))
            ))
        if not rows: return
        with self.engine.begin() as con:
            stmt = pg_insert(self.events).values(rows).on_conflict_do_nothing(constraint="uq_event_unique")
            con.execute(stmt)
            
    def create_patient(self, patient_data: Dict) -> Dict:
        with self.engine.begin() as con:
            stmt = insert(self.patients).values(**patient_data).returning(self.patients)
            result = con.execute(stmt).fetchone()
            return result._asdict() if result else None
            
    def get_patient_by_id(self, patient_id: str) -> Optional[Dict]:
        with self.engine.begin() as con:
            stmt = select(self.patients).where(self.patients.c.patient_id == patient_id)
            result = con.execute(stmt).fetchone()
            return result._asdict() if result else None

    def get_patients(self, skip: int = 0, limit: int = 100) -> List[Dict]:
        with self.engine.begin() as con:
            stmt = select(self.patients).offset(skip).limit(limit)
            results = con.execute(stmt).fetchall()
            return [r._asdict() for r in results]
            
    def update_patient(self, patient_id: str, update_data: Dict) -> Optional[Dict]:
        with self.engine.begin() as con:
            stmt = (
                update(self.patients)
                .where(self.patients.c.patient_id == patient_id)
                .values(**update_data)
                .returning(self.patients)
            )
            result = con.execute(stmt).fetchone()
            return result._asdict() if result else None

    def delete_patient(self, patient_id: str) -> bool:
        with self.engine.begin() as con:
            stmt = delete(self.patients).where(self.patients.c.patient_id == patient_id)
            result = con.execute(stmt)
            return result.rowcount > 0

    # ... (load_raw_points and other load methods remain the same)
    def load_raw_points(self, session_id: str, since: datetime, until: datetime) -> List[Dict]:
        with self.engine.begin() as con:
            res = con.execute(
                text("""
                    SELECT ts, bpm, ua FROM ctg_samples
                    WHERE session_id=:sid AND ts BETWEEN :since AND :until
                    ORDER BY ts
                """),
                dict(sid=session_id, since=since, until=until),
            )
            return [{"ts": r[0], "bpm": r[1], "ua": r[2]} for r in res]

    def load_events(self, session_id: str, since: Optional[datetime], until: Optional[datetime]) -> List[Dict]:
        q = "SELECT type, start_ts, end_ts, dur_s, min_bpm, max_drop, severity FROM events WHERE session_id=:sid"
        if since: q += " AND (end_ts IS NULL OR end_ts >= :since)"
        if until: q += " AND start_ts <= :until"
        q += " ORDER BY start_ts"
        with self.engine.begin() as con:
            res = con.execute(text(q), dict(sid=session_id, since=since, until=until))
            return [dict(type=r[0], start_ts=r[1], end_ts=r[2], dur_s=r[3], min_bpm=r[4], max_drop=r[5], severity=r[6]) for r in res]

    def load_risk(self, session_id: str, limit: int = 200) -> List[Dict]:
        with self.engine.begin() as con:
            res = con.execute(
                text("""
                    SELECT ts, hypoxia_prob, band FROM risk_records
                    WHERE session_id=:sid ORDER BY ts DESC LIMIT :lim
                """),
                dict(sid=session_id, lim=limit)
            )
            return [dict(ts=r[0], hypoxia_prob=r[1], band=r[2]) for r in res]

def _severity_from_drop(max_drop: float, dur_s: float) -> str:
    if max_drop is None: return "unknown"
    if max_drop >= 25 and dur_s >= 10: return "severe"
    if dur_s >= 120: return "prolonged"
    return "light"