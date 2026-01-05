from sqlalchemy import Column, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from database import Base
from datetime import datetime

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(Text)
    raw_text = Column(Text)
    
    # Simple Analysis JSON (For chat titles) - No vectors here
    analysis = Column(JSONB, nullable=True) 
    
    uploaded_at = Column(TIMESTAMP, default=datetime.utcnow)