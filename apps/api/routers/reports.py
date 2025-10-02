from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone

from ..deps import get_storage
from packages.ctg_core.schemas import SessionReport, Patient
from ..auth import get_current_user, TokenData

router = APIRouter(prefix="/v1/reports", tags=["reports"])

@router.get("/sessions/{session_id}", response_model=SessionReport)
def get_session_report(
    session_id: str,
    storage = Depends(get_storage),
    current_user: TokenData = Depends(get_current_user),
):
    """Generate comprehensive summary report for a session"""
    report_data = storage.get_session_report_data(session_id)
    if not report_data:
        raise HTTPException(
            status_code=404, 
            detail="Session not found or contains no data"
        )

    session_info = report_data["session_info"]
    time_bounds = report_data["time_bounds"]
    
    patient_info = None
    if session_info.get("patients_patient_id"):
        patient_info = Patient(
            patient_id=session_info["patients_patient_id"],
            medical_record_number=session_info["patients_medical_record_number"],
            full_name=session_info["patients_full_name"],
            date_of_birth=session_info["patients_date_of_birth"],
            created_at=session_info["patients_created_at"],
            updated_at=session_info["patients_updated_at"]
        )

    report = SessionReport(
        session_id=session_id,
        patient_id=session_info.get("sessions_patient_id"),
        patient_info=patient_info,
        session_start_time=time_bounds.min_ts if time_bounds else None,
        session_end_time=time_bounds.max_ts if time_bounds else None,
        generated_at=datetime.now(timezone.utc),
        risk_summary=report_data["risk_summary"],
        event_summary=report_data["event_summary"],
        signal_summary={
            "total_duration_minutes": report_data["total_duration_minutes"]
        }
    )

    return report