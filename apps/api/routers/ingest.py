from fastapi import APIRouter, HTTPException, Depends
from packages.ctg_core.schemas import IngestBatch, RiskOutput
from ..deps import get_storage, get_streaming_service
from datetime import datetime, timezone
from .adapters import replace_nan_with_none

router = APIRouter(prefix="/v1", tags=["ingest"])

@router.post("/ingest/batch")
def ingest_batch(
    batch: IngestBatch,
    storage = Depends(get_storage),
    streaming = Depends(get_streaming_service)
):
    # Сохранить в БД
    storage.upsert_session(batch.session_id, batch.device_id, batch.patient_id)
    storage.insert_samples(batch.session_id, [s.model_dump() for s in batch.samples])
    
    # Онлайновая обработка
    streaming.ingest(batch.session_id, [s.model_dump() for s in batch.samples])
    result = streaming.tick(batch.session_id)
    
    if result and result.get("risk"):
        storage.save_risk(
            batch.session_id, 
            datetime.now(timezone.utc), 
            result["risk"]["hypoxia_prob"], 
            result["risk"]["band"]
        )
    
    if result and result.get("decel_events"):
        storage.save_new_decel_events(batch.session_id, result["decel_events"])
    
    return {"ok": True, "result": replace_nan_with_none(result)}

@router.get("/sessions/{session_id}/risk", response_model=RiskOutput)
def get_risk(
    session_id: str,
    storage = Depends(get_storage),
    streaming = Depends(get_streaming_service)
):
    result = streaming.tick(session_id)
    
    if result is None or "risk" not in result:
        hist = storage.load_risk(session_id, limit=1)
        if not hist:
            raise HTTPException(status_code=404, detail="No risk yet")
        r = hist[0]
        return RiskOutput(
            session_id=session_id, 
            hypoxia_prob=r["hypoxia_prob"], 
            band=r["band"], 
            updated_at=r["ts"]
        )
    
    r = result["risk"]
    return RiskOutput(
        session_id=session_id, 
        hypoxia_prob=r["hypoxia_prob"], 
        band=r["band"], 
        updated_at=result["ts"]
    )