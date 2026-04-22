"""
Face detection and embedding pipeline using facenet-pytorch.

Pipeline:
    1. Decode image (bytes → PIL)
    2. Validate dimensions and file size
    3. Face detection via MTCNN → bounding boxes + confidence + landmarks
    4. Crop & align primary face
    5. Embedding extraction via InceptionResnetV1
    6. L2-normalize embedding
    7. Return structured result for storage or comparison
"""

import io
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes for structured pipeline output
# ---------------------------------------------------------------------------

@dataclass
class FaceBox:
    """Single detected face bounding box."""
    box: list[float]          # [x1, y1, x2, y2]
    confidence: float
    landmarks: dict | None = None  # left_eye, right_eye, nose, mouth_left, mouth_right


@dataclass
class DetectionResult:
    """Full detection response."""
    faces: list[FaceBox]
    image_width: int
    image_height: int


@dataclass
class EmbeddingResult:
    """Embedding extraction result for a single face."""
    embedding: np.ndarray     # 512-d normalized vector
    quality_score: float      # detection confidence as quality proxy
    box: list[float]
    latency_ms: float


@dataclass
class VerifyResult:
    """1:1 verification result."""
    score: float
    threshold: float
    match: bool
    latency_ms: float


@dataclass
class RecognizeMatch:
    """Single recognition match."""
    subject_id: str
    score: float


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

class FacePipeline:
    """
    Face detection + embedding pipeline.

    Models are loaded once at construction time and kept on the selected
    device (CPU or CUDA) for the lifetime of the service.
    """

    def __init__(self):
        settings = get_settings()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "face_pipeline.init",
            device=str(self._device),
            pretrained=settings.face_model_pretrained,
        )

        # MTCNN detector
        self._mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=settings.face_min_face_size,
            thresholds=[0.6, 0.7, settings.face_detect_threshold],
            factor=0.709,
            keep_all=True,
            device=self._device,
            post_process=True,  # normalize to [-1, 1]
        )

        # InceptionResnetV1 embedder
        self._embedder = InceptionResnetV1(
            pretrained=settings.face_model_pretrained,
            device=self._device,
        ).eval()

        self._settings = settings
        logger.info("face_pipeline.ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_bytes: bytes) -> DetectionResult:
        """
        Detect all faces in the given image.

        Returns bounding boxes, confidence scores, and facial landmarks.
        """
        img = self._decode_image(image_bytes)
        w, h = img.size

        boxes, probs, landmarks = self._mtcnn.detect(img, landmarks=True)

        faces: list[FaceBox] = []
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                lm = None
                if landmarks is not None and i < len(landmarks):
                    pts = landmarks[i]
                    lm = {
                        "left_eye": [float(pts[0][0]), float(pts[0][1])],
                        "right_eye": [float(pts[1][0]), float(pts[1][1])],
                        "nose": [float(pts[2][0]), float(pts[2][1])],
                        "mouth_left": [float(pts[3][0]), float(pts[3][1])],
                        "mouth_right": [float(pts[4][0]), float(pts[4][1])],
                    }
                faces.append(FaceBox(
                    box=[float(c) for c in box],
                    confidence=float(prob),
                    landmarks=lm,
                ))

        return DetectionResult(faces=faces, image_width=w, image_height=h)

    def extract_embedding(self, image_bytes: bytes) -> EmbeddingResult:
        """
        Detect the primary face and extract a 512-d embedding.

        Raises ValueError if zero or multiple faces are detected
        (controlled by FACE_MAX_FACES_ENROLL).
        """
        t0 = time.perf_counter()
        img = self._decode_image(image_bytes)

        # Detect
        boxes, probs, _ = self._mtcnn.detect(img, landmarks=False)

        if boxes is None or len(boxes) == 0:
            raise ValueError("No face detected in the image.")

        if len(boxes) > self._settings.face_max_faces_enroll:
            raise ValueError(
                f"Multiple faces detected ({len(boxes)}). "
                f"Maximum allowed: {self._settings.face_max_faces_enroll}."
            )

        # Pick highest-confidence face
        best_idx = int(np.argmax(probs))
        best_box = boxes[best_idx]
        best_prob = float(probs[best_idx])

        if best_prob < self._settings.face_detect_threshold:
            raise ValueError(
                f"Face detection confidence too low: {best_prob:.3f} "
                f"(threshold: {self._settings.face_detect_threshold})."
            )

        # Crop & align via MTCNN (returns tensor [1, 3, 160, 160])
        face_tensor = self._mtcnn(img)
        if face_tensor is None:
            raise ValueError("Failed to crop and align face.")

        # If keep_all=True, select the best face tensor
        if face_tensor.dim() == 4:
            face_tensor = face_tensor[best_idx].unsqueeze(0)

        # Extract embedding
        face_tensor = face_tensor.to(self._device)
        with torch.no_grad():
            emb = self._embedder(face_tensor)

        # L2 normalize
        emb_np = emb.cpu().numpy().flatten()
        emb_np = emb_np / np.linalg.norm(emb_np)

        latency = (time.perf_counter() - t0) * 1000

        return EmbeddingResult(
            embedding=emb_np,
            quality_score=best_prob,
            box=[float(c) for c in best_box],
            latency_ms=round(latency, 2),
        )

    def verify(
        self,
        image_bytes: bytes,
        stored_embeddings: list[np.ndarray],
        threshold: float | None = None,
    ) -> VerifyResult:
        """
        1:1 verification — compare an image against stored embeddings.

        Uses the maximum cosine similarity across all stored templates.
        """
        t0 = time.perf_counter()
        thr = threshold or self._settings.face_verify_threshold

        result = self.extract_embedding(image_bytes)
        probe = result.embedding

        # Cosine similarity against each stored embedding
        best_score = -1.0
        for stored in stored_embeddings:
            stored_norm = stored / np.linalg.norm(stored)
            score = float(np.dot(probe, stored_norm))
            if score > best_score:
                best_score = score

        latency = (time.perf_counter() - t0) * 1000

        return VerifyResult(
            score=round(best_score, 4),
            threshold=thr,
            match=best_score >= thr,
            latency_ms=round(latency, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_image(self, image_bytes: bytes) -> Image.Image:
        """Decode raw bytes to a PIL RGB image with size validation."""
        if len(image_bytes) > self._settings.max_image_size:
            raise ValueError(
                f"Image size ({len(image_bytes)} bytes) exceeds "
                f"maximum ({self._settings.max_image_size} bytes)."
            )
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid image data: {exc}") from exc
        return img


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_pipeline: FacePipeline | None = None


def get_pipeline() -> FacePipeline:
    """Return the global FacePipeline singleton (lazy-init)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FacePipeline()
    return _pipeline


def init_pipeline() -> None:
    """Eagerly initialize the pipeline at startup."""
    get_pipeline()
