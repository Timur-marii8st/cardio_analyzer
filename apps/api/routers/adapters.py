from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List
from ..services.adapters import parse_csv_to_samples, parse_fhir_observations
from ..deps import get_streaming_service
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["adapters"])

class FolderIngestRequest(BaseModel):
    session_id: str
    folder_name: str

@router.post("/ingest/csv")
async def ingest_csv(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
    channel: str = Form(...),
    streaming = Depends(get_streaming_service)
):
    """
    Загрузка одного или нескольких CSV файлов с данными CTG для одного канала.
    """
    all_samples = []
    for file in files:
        content = await file.read()
        try:
            samples = parse_csv_to_samples(content, session_id=session_id, channel=channel)
            if samples:
                all_samples.extend(samples)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"CSV parse error in file '{file.filename}': {e}"
            )
    if not all_samples:
        raise HTTPException(status_code=400, detail="No valid samples found in the provided files")
    streaming.ingest(session_id, all_samples)
    result = streaming.tick(session_id)
    return {
        "ok": True,
        "files_processed": len(files),
        "total_samples_ingested": len(all_samples),
        "channel": channel,
        "result": result
    }

@router.post("/ingest/folder")
async def ingest_folder(
    payload: FolderIngestRequest,
    streaming = Depends(get_streaming_service)
):
    session_id = payload.session_id
    folder_name = payload.folder_name
    
    try:
        samples = parse_csv_to_samples(folder_name, session_id=session_id)
        streaming.ingest(session_id, samples)
        result = streaming.tick(session_id)
        return {"ok": True, "result": result}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Folder ingest error: {e}")

@router.post("/ingest/fhir/observation")
async def ingest_fhir_observation(
    session_id: str,
    payload: dict,
    streaming = Depends(get_streaming_service)
):
    try:
        samples = parse_fhir_observations(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"FHIR parse error: {e}")
    
    if not samples:
        raise HTTPException(status_code=400, detail="No samples found in Observation")
    
    streaming.ingest(session_id, samples)
    result = streaming.tick(session_id)
    return {"ok": True, "ingested": len(samples), "result": result}