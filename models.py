# models.py
from typing import Optional
from sqlmodel import SQLModel, Field

class SkinCancerScan(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image: bytes = Field(..., description="Raw image data")
    scan_result: str = Field(..., description="Scan result stored as a JSON string")
    actual_result: Optional[str] = Field(default=None, nullable=True, description="Actual scan result stored as a JSON string")
    
