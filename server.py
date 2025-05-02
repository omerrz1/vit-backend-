# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from io import BytesIO
import json
from PIL import Image
from sqlmodel import Session, select
from contextlib import asynccontextmanager

from db import engine, create_db_and_tables
from models import SkinCancerScan
from scan import scanImage

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create database tables
    create_db_and_tables()
    yield
    # Shutdown: (Optional cleanup can be performed here.)

app = FastAPI(lifespan=lifespan)
from fastapi.staticfiles import StaticFiles
import os

# Create the explainability folder if it doesn't exist
os.makedirs("explainability", exist_ok=True)

# Mount the folder for static file access
app.mount("/explainability", StaticFiles(directory="explainability"), name="explainability")

# Dependency to get a session
def get_session():
    with Session(engine) as session:
        yield session

@app.post("/quick-scan/")
async def quick_scan(
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    raw_data = await file.read()
    file_bytes = bytes(raw_data)

    image_stream = BytesIO(file_bytes)
    img = Image.open(image_stream).convert("RGB")

    # Run scan
    result = scanImage(img)

    # Save to DB
    result_json = json.dumps(result["results"])
    explainability_image_path = result["explainability_image"]  # Make sure this is a path relative to your static folder.

    scan = SkinCancerScan(image=file_bytes, scan_result=result_json, actual_result=None)
    session.add(scan)
    session.commit()
    session.refresh(scan)

    # Return scan + path to image
    return {
        "id": scan.id,
        "result": result["results"],
        "explainability_image_path": explainability_image_path,  # Make sure it's returned here
    }


@app.get("/scans/")
def list_scans(session: Session = Depends(get_session)):
    scans = session.exec(select(SkinCancerScan)).all()
    return scans


@app.put("/update-result/{scan_id}/")
async def update_actual_result(
    scan_id: int,
    actual_result: str = Form(...),
    session: Session = Depends(get_session)
):
    # Find the scan entry
    scan = session.get(SkinCancerScan, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Update the actual result
    scan.actual_result = actual_result
    session.add(scan)
    session.commit()
    session.refresh(scan)

    return {"message": "Actual result updated successfully", "scan_id": scan.id, "ActualResult": scan.actual_result}


