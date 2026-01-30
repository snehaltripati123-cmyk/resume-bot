import uuid
from datetime import datetime
from sqlalchemy import Column, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from database import Base

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(Text)
    raw_text = Column(Text)
    
    # --- MISSING COLUMNS FIXED HERE ---
    embedding = Column(Vector(768)) 
    
    # This is the one causing your error:
    is_processed = Column(Integer, default=1) 
    
    analysis = Column(JSONB, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class ComparisonSession(Base):
    __tablename__ = "comparison_sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    candidates = relationship("ComparisonCandidate", back_populates="session")

class ComparisonCandidate(Base):
    __tablename__ = "comparison_candidates"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    comparison_id = Column(UUID(as_uuid=True), ForeignKey("comparison_sessions.id"))
    resume_id = Column(UUID(as_uuid=True), ForeignKey("resumes.id"))
    added_at = Column(DateTime, default=datetime.utcnow)
    
    resume = relationship("Resume")
    session = relationship("ComparisonSession", back_populates="candidates")