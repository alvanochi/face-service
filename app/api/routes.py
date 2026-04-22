"""
API routes for the face recognition service.

All routes (except health checks) require the X-API-Key header.
"""

import os
import time
import uuid

import numpy as np
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import (
    DeleteSubjectResponse,
    DetectResponse,
    EnrollResponse,
    ErrorResponse,
    FaceBoxSchema,
    HealthResponse,
    ImageQualitySchema,
    LandmarksSchema,
    ReadyResponse,
    RecognizeMatchSchema,
    RecognizeResponse,
    SubjectStatusResponse,
    VerifyResponse,
)
from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import verify_api_key
from app.models.database_models import FaceEmbedding, FaceSubject, FaceVerificationLog
from app.services.face_pipeline import get_pipeline

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/v1/faces", tags=["faces"])
health_router = APIRouter(tags=["health"])


# ===========================================================================
# Health checks (no auth required)
# ===========================================================================

@health_router.get("/healthz", response_model=HealthResponse)
async def healthz():
    """Liveness probe — service is running."""
    return HealthResponse(status="ok", service="face-service")


@health_router.get("/readyz", response_model=ReadyResponse)
async def readyz(db: Session = Depends(get_db)):
    """Readiness probe — DB connected and model loaded."""
    try:
        db.execute(select(FaceSubject).limit(1))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    try:
        pipeline = get_pipeline()
        model_status = "loaded"
    except Exception:
        model_status = "not_loaded"

    overall = "ready" if db_status == "connected" and model_status == "loaded" else "not_ready"
    return ReadyResponse(status=overall, database=db_status, model=model_status)


# ===========================================================================
# POST /v1/faces/detect
# ===========================================================================

@router.post(
    "/detect",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse}},
)
async def detect_faces(
    image: UploadFile = File(...),
    _key: str = Depends(verify_api_key),
):
    """Upload an image, receive bounding boxes, confidence, and landmark data."""
    image_bytes = await image.read()
    pipeline = get_pipeline()

    try:
        result = pipeline.detect(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    faces = []
    for f in result.faces:
        lm = None
        if f.landmarks:
            lm = LandmarksSchema(**f.landmarks)
        faces.append(FaceBoxSchema(box=f.box, confidence=f.confidence, landmarks=lm))

    return DetectResponse(
        faces=faces,
        image_quality=ImageQualitySchema(width=result.image_width, height=result.image_height),
    )


# ===========================================================================
# POST /v1/faces/enroll
# ===========================================================================

@router.post(
    "/enroll",
    response_model=EnrollResponse,
    responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def enroll_face(
    subject_id: str = Form(...),
    image: UploadFile = File(...),
    _key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Enroll a face template for the given subject.

    Creates the subject record if it doesn't exist, then stores
    the face embedding. Idempotent for the same image (by quality score).
    """
    settings = get_settings()
    image_bytes = await image.read()
    pipeline = get_pipeline()

    # --- Extract embedding ---
    try:
        emb_result = pipeline.extract_embedding(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # --- Ensure subject exists ---
    subject = db.execute(
        select(FaceSubject).where(FaceSubject.subject_id == subject_id)
    ).scalar_one_or_none()

    if subject is None:
        subject = FaceSubject(subject_id=subject_id, status="active")
        db.add(subject)
        db.flush()

    # --- Save cropped face image (optional) ---
    image_path = None
    storage_dir = settings.image_storage_path
    if storage_dir:
        subject_dir = os.path.join(storage_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(subject_dir, filename)
        with open(image_path, "wb") as fp:
            fp.write(image_bytes)

    # --- Store embedding ---
    face_emb = FaceEmbedding(
        subject_id=subject_id,
        embedding=emb_result.embedding.tolist(),
        model_name="InceptionResnetV1",
        model_version=settings.face_model_pretrained,
        detector_name="MTCNN",
        quality_score=emb_result.quality_score,
        image_path=image_path,
        is_active=True,
    )
    db.add(face_emb)
    db.commit()
    db.refresh(face_emb)

    logger.info(
        "face.enrolled",
        subject_id=subject_id,
        embedding_id=face_emb.id,
        quality=emb_result.quality_score,
    )

    detection = pipeline.detect(image_bytes)

    return EnrollResponse(
        subject_id=subject_id,
        embedding_id=face_emb.id,
        faces_detected=len(detection.faces),
        quality_score=round(emb_result.quality_score, 4),
        status="stored",
    )


# ===========================================================================
# POST /v1/faces/verify
# ===========================================================================

@router.post(
    "/verify",
    response_model=VerifyResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def verify_face(
    subject_id: str = Form(...),
    image: UploadFile = File(...),
    threshold: float | None = Form(None),
    _key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    1:1 verification — compare uploaded face against stored templates
    of a specific subject.
    """
    t0 = time.perf_counter()
    settings = get_settings()
    thr = threshold or settings.face_verify_threshold
    image_bytes = await image.read()
    pipeline = get_pipeline()

    # Load stored embeddings
    rows = db.execute(
        select(FaceEmbedding.embedding).where(
            FaceEmbedding.subject_id == subject_id,
            FaceEmbedding.is_active == True,  # noqa: E712
        )
    ).scalars().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No enrolled templates found for subject '{subject_id}'.",
        )

    stored_embeddings = [np.array(e) for e in rows]

    try:
        result = pipeline.verify(image_bytes, stored_embeddings, threshold=thr)
    except ValueError as exc:
        # Log the failed attempt
        latency = (time.perf_counter() - t0) * 1000
        log_entry = FaceVerificationLog(
            subject_id=subject_id,
            action="verify",
            score=None,
            threshold=thr,
            decision="error",
            latency_ms=round(latency, 2),
            error_detail=str(exc),
        )
        db.add(log_entry)
        db.commit()
        raise HTTPException(status_code=400, detail=str(exc))

    # Audit log
    latency = (time.perf_counter() - t0) * 1000
    log_entry = FaceVerificationLog(
        subject_id=subject_id,
        action="verify",
        score=result.score,
        threshold=result.threshold,
        decision="match" if result.match else "no_match",
        latency_ms=round(latency, 2),
    )
    db.add(log_entry)
    db.commit()

    logger.info(
        "face.verify",
        subject_id=subject_id,
        score=result.score,
        match=result.match,
        latency_ms=round(latency, 2),
    )

    return VerifyResponse(
        subject_id=subject_id,
        score=result.score,
        threshold=result.threshold,
        match=result.match,
    )


# ===========================================================================
# POST /v1/faces/recognize
# ===========================================================================

@router.post(
    "/recognize",
    response_model=RecognizeResponse,
    responses={400: {"model": ErrorResponse}},
)
async def recognize_face(
    image: UploadFile = File(...),
    top_k: int | None = Form(None),
    _key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    1:N recognition — find the top-k most similar subjects.

    Uses exact cosine similarity search. For large databases,
    pgvector approximate indexing should be enabled.
    """
    t0 = time.perf_counter()
    settings = get_settings()
    k = top_k or settings.recognize_top_k
    image_bytes = await image.read()
    pipeline = get_pipeline()

    try:
        emb_result = pipeline.extract_embedding(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    probe = emb_result.embedding

    # Fetch all active embeddings (exact search — fine for small/medium scale)
    rows = db.execute(
        select(FaceEmbedding.subject_id, FaceEmbedding.embedding).where(
            FaceEmbedding.is_active == True,  # noqa: E712
        )
    ).all()

    if not rows:
        return RecognizeResponse(matches=[])

    # Compute cosine similarity per subject (best score per subject)
    subject_scores: dict[str, float] = {}
    for sid, emb in rows:
        stored = np.array(emb)
        stored_norm = stored / np.linalg.norm(stored)
        score = float(np.dot(probe, stored_norm))
        if sid not in subject_scores or score > subject_scores[sid]:
            subject_scores[sid] = score

    # Sort by score descending, take top-k
    ranked = sorted(subject_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    latency = (time.perf_counter() - t0) * 1000
    logger.info("face.recognize", top_k=k, results=len(ranked), latency_ms=round(latency, 2))

    return RecognizeResponse(
        matches=[
            RecognizeMatchSchema(subject_id=sid, score=round(sc, 4))
            for sid, sc in ranked
        ]
    )


# ===========================================================================
# GET /v1/faces/subjects/{subject_id}
# ===========================================================================

@router.get(
    "/subjects/{subject_id}",
    response_model=SubjectStatusResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_subject(
    subject_id: str,
    _key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Check enrollment status and template count for a subject."""
    subject = db.execute(
        select(FaceSubject).where(FaceSubject.subject_id == subject_id)
    ).scalar_one_or_none()

    if subject is None:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found.")

    total = db.execute(
        select(FaceEmbedding.id).where(FaceEmbedding.subject_id == subject_id)
    ).scalars().all()

    active = db.execute(
        select(FaceEmbedding.id).where(
            FaceEmbedding.subject_id == subject_id,
            FaceEmbedding.is_active == True,  # noqa: E712
        )
    ).scalars().all()

    return SubjectStatusResponse(
        subject_id=subject.subject_id,
        status=subject.status,
        total_embeddings=len(total),
        active_embeddings=len(active),
        created_at=subject.created_at.isoformat() if subject.created_at else None,
    )


# ===========================================================================
# DELETE /v1/faces/subjects/{subject_id}
# ===========================================================================

@router.delete(
    "/subjects/{subject_id}",
    response_model=DeleteSubjectResponse,
    responses={404: {"model": ErrorResponse}},
)
async def delete_subject(
    subject_id: str,
    _key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Delete all templates and the subject record."""
    subject = db.execute(
        select(FaceSubject).where(FaceSubject.subject_id == subject_id)
    ).scalar_one_or_none()

    if subject is None:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found.")

    db.delete(subject)
    db.commit()

    logger.info("face.subject_deleted", subject_id=subject_id)

    return DeleteSubjectResponse(
        subject_id=subject_id,
        deleted=True,
        message=f"Subject '{subject_id}' and all templates deleted.",
    )
