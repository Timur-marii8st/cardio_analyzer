from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List, Optional
from datetime import datetime, date

# --- Core Signal Schemas ---

Channel = Literal["bpm", "uterus"]

class Sample(BaseModel):
    ts: datetime
    channel: Channel
    value: float
    quality: Optional[int] = Field(default=None)
    model_config = ConfigDict(extra="ignore")

class IngestBatch(BaseModel):
    device_id: str
    session_id: str
    patient_id: Optional[str] = None
    samples: List[Sample]

class RiskOutput(BaseModel):
    session_id: str
    hypoxia_prob: float
    band: Literal["normal", "elevated", "high"]
    updated_at: datetime

class Event(BaseModel):
    type: Literal["deceleration", "acceleration", "contraction", "tachycardia", "bradycardia", "low_variability"]
    start: datetime
    end: Optional[datetime] = None
    details: dict = Field(default_factory=dict)

# --- Patient Schemas ---

class PatientBase(BaseModel):
    full_name: str
    medical_record_number: str
    date_of_birth: Optional[date] = None

class PatientCreate(PatientBase):
    pass

class PatientUpdate(PatientBase):
    full_name: Optional[str] = None
    medical_record_number: Optional[str] = None
    date_of_birth: Optional[date] = None

class Patient(PatientBase):
    patient_id: str
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

class ReportRiskSummary(BaseModel):
    mean_hypoxia_prob: Optional[float] = None
    max_hypoxia_prob: Optional[float] = None
    minutes_in_normal_band: float
    minutes_in_elevated_band: float
    minutes_in_high_band: float

class ReportEventSummary(BaseModel):
    total_decelerations: int
    light_decelerations: int
    severe_decelerations: int
    prolonged_decelerations: int
    # Placeholders for future event types
    total_accelerations: int = 0
    total_contractions: int = 0

class ReportSignalSummary(BaseModel):
    total_duration_minutes: float
    # Placeholder for future quality metrics
    signal_loss_percentage: Optional[float] = None

class SessionReport(BaseModel):
    session_id: str
    patient_id: Optional[str] = None
    patient_info: Optional[Patient] = None
    session_start_time: Optional[datetime] = None
    session_end_time: Optional[datetime] = None
    generated_at: datetime
    risk_summary: ReportRiskSummary
    event_summary: ReportEventSummary
    signal_summary: ReportSignalSummary

class MLSettings(BaseModel):
    # Common
    target_freq_hz: float = 4.0

    # Anomalies
    tachy_bpm: int = 160
    brady_bpm: int = 110
    decel_light_drop_bpm: int = 15
    decel_severe_drop_bpm: int = 25
    decel_prolonged_sec: int = 120
    decel_min_sec: int = 10
    low_var_stv: float = 1.0
    acc_rise_bpm: int = 10
    acc_min_sec: int = 10      
    ua_contr_min_rise: int = 10
    ua_contr_min_sec: int = 20 

    # Features
    baseline_roll_sec: int = 60
    baseline_event_roll_sec: int = 30

    # Processing
    bpm_range_lo: int = 50
    bpm_range_hi: int = 220
    ua_range_lo: int = 0
    ua_range_hi: int = 120
    resample_every: str = "250ms"

    # Realtime
    feature_window_sec: int = 600
    min_window_points: int = 1200

    class Config:
        extra = "forbid"

ml_settings = MLSettings()