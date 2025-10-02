from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List

from ..deps import get_storage
from packages.ctg_core.schemas import Patient, PatientCreate, PatientUpdate
from ..auth import get_current_user, TokenData

router = APIRouter(prefix="/v1/patients", tags=["patients"])

@router.post("/", response_model=Patient, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient: PatientCreate,
    storage = Depends(get_storage),
    current_user: TokenData = Depends(get_current_user)
):
    """Create a new patient record."""
    try:
        created_patient = storage.create_patient(patient.model_dump())
        if not created_patient:
            raise HTTPException(status_code=400, detail="Patient could not be created.")
        return created_patient
    except Exception as e:
        raise HTTPException(
            status_code=409, 
            detail=f"Patient with this medical record number may already exist: {e}"
        )

@router.get("/", response_model=List[Patient])
def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    storage = Depends(get_storage),
    current_user: TokenData = Depends(get_current_user)
):
    """Retrieve a list of patients with pagination."""
    patients = storage.get_patients(skip=skip, limit=limit)
    return patients

@router.get("/{patient_id}", response_model=Patient)
def get_patient(
    patient_id: str,
    storage = Depends(get_storage),
    current_user: TokenData = Depends(get_current_user)
):
    """Retrieve a single patient by their ID."""
    patient = storage.get_patient_by_id(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.put("/{patient_id}", response_model=Patient)
def update_patient(
    patient_id: str,
    patient_update: PatientUpdate,
    storage = Depends(get_storage),
    current_user: TokenData = Depends(get_current_user)
):
    """Update a patient's details."""
    update_data = patient_update.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided.")
    
    updated_patient = storage.update_patient(patient_id, update_data)
    if updated_patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return updated_patient

@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: str,
    storage = Depends(get_storage),
    current_user: TokenData = Depends(get_current_user)
):
    """Delete a patient record."""
    success = storage.delete_patient(patient_id)
    if not success:
        raise HTTPException(status_code=404, detail="Patient not found")
    return None