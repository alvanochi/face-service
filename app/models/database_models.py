"""
SQLAlchemy ORM models for face recognition data.

Tables:
    face_subjects          — enrolled user/subject records
    face_embeddings        — face embedding vectors per subject
    face_verification_logs — audit trail of every verify/recognize decision
"""

import datetime
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.core.database import Base


class FaceSubject(Base):
    """
    A subject (person) enrolled in the face recognition system.

    subject_id corresponds to the user id / identifier in the Laravel app.
    """

    __tablename__ = "face_subjects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String(100), unique=True, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="active")  # active | inactive | suspended
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    embeddings = relationship(
        "FaceEmbedding",
        back_populates="subject",
        cascade="all, delete-orphan",
    )
    verification_logs = relationship(
        "FaceVerificationLog",
        back_populates="subject",
        cascade="all, delete-orphan",
    )


class FaceEmbedding(Base):
    """
    A single face embedding vector for a subject.

    Multiple embeddings per subject are encouraged (3–5 from different
    angles/lighting) for better matching accuracy.
    """

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(
        String(100),
        ForeignKey("face_subjects.subject_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 512-dimensional embedding from InceptionResnetV1
    embedding = Column(Vector(512), nullable=False)

    model_name = Column(String(100), nullable=False, default="InceptionResnetV1")
    model_version = Column(String(50), nullable=False, default="vggface2")
    detector_name = Column(String(50), nullable=False, default="MTCNN")
    quality_score = Column(Float, nullable=True)
    image_path = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    subject = relationship("FaceSubject", back_populates="embeddings")

    __table_args__ = (
        Index("ix_face_embeddings_active", "subject_id", "is_active"),
    )


class FaceVerificationLog(Base):
    """
    Immutable audit log for every verification / recognition decision.

    Records the score, threshold, and pass/fail decision along with
    timing information for SLA monitoring.
    """

    __tablename__ = "face_verification_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(
        String(100),
        ForeignKey("face_subjects.subject_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    request_id = Column(String(100), nullable=True, index=True)
    action = Column(String(20), nullable=False, default="verify")  # verify | recognize
    score = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    decision = Column(String(20), nullable=False)  # match | no_match | error
    latency_ms = Column(Float, nullable=True)
    source = Column(String(100), nullable=True)  # e.g. "laravel-auth", "mobile"
    error_detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    subject = relationship("FaceSubject", back_populates="verification_logs")
