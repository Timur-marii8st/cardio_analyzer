from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List, Optional, Dict, Any
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

class SessionReport(BaseModel):
    session_id: str
    patient_id: Optional[str] = None
    patient_info: Optional["Patient"] = None  # ссылка на модель Patient
    session_start_time: Optional[datetime] = None
    session_end_time: Optional[datetime] = None
    generated_at: datetime
    risk_summary: Dict[str, Any]
    event_summary: Dict[str, Any]
    signal_summary: Dict[str, Any]