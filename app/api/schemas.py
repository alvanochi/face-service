"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class LandmarksSchema(BaseModel):
    left_eye: list[float]
    right_eye: list[float]
    nose: list[float] | None = None
    mouth_left: list[float] | None = None
    mouth_right: list[float] | None = None


class FaceBoxSchema(BaseModel):
    box: list[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    confidence: float
    landmarks: LandmarksSchema | None = None


class ImageQualitySchema(BaseModel):
    width: int
    height: int


class DetectResponse(BaseModel):
    faces: list[FaceBoxSchema]
    image_quality: ImageQualitySchema


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

class EnrollResponse(BaseModel):
    subject_id: str
    embedding_id: int
    faces_detected: int
    quality_score: float
    status: str


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class VerifyResponse(BaseModel):
    subject_id: str
    score: float
    threshold: float
    match: bool


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

class RecognizeMatchSchema(BaseModel):
    subject_id: str
    score: float


class RecognizeResponse(BaseModel):
    matches: list[RecognizeMatchSchema]


# ---------------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------------

class SubjectStatusResponse(BaseModel):
    subject_id: str
    status: str
    total_embeddings: int
    active_embeddings: int
    created_at: str | None = None


class DeleteSubjectResponse(BaseModel):
    subject_id: str
    deleted: bool
    message: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str = "1.0.0"


class ReadyResponse(BaseModel):
    status: str
    database: str
    model: str


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
