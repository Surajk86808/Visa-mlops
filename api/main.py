from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import os
import glob
import pandas as pd

from Visa.entity.estimator import USvisaModel, TargetValueMapping
from Visa.utils.main_utils import load_object
from Visa.pipline.training_pipeline import TrainingPipeline

app = FastAPI(title="US Visa MLOps API")

# Enable CORS for cross-origin requests (configure origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your Vercel domain(s) for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


class VisaInput(BaseModel):
    # Raw input fields expected by transformation
    case_id: Optional[str] = None
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    no_of_employees: int
    yr_of_estab: int
    region_of_employment: str
    prevailing_wage: int
    unit_of_wage: str
    full_time_position: str


class VisaBatchRequest(BaseModel):
    items: List[VisaInput]


MODEL_FILE_NAME = "model.pkl"
ARTIFACT_ROOT = "artifact"


def _find_latest_model_path() -> Optional[str]:
    # Finds latest trained model file under artifact/*/model_trainer/trained_model/model.pkl
    pattern = os.path.join(ARTIFACT_ROOT, "*", "model_trainer", "trained_model", MODEL_FILE_NAME)
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # Extract timestamp folder and sort descending by timestamp name (assumes sortable format)
    candidates.sort(key=lambda p: os.path.normpath(p).split(os.sep)[1], reverse=True)
    return candidates[0]


_cached_model: Optional[USvisaModel] = None


def _load_model() -> USvisaModel:
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    model_path = _find_latest_model_path()
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Please run training first.")
    _cached_model = load_object(model_path)
    return _cached_model


@app.get("/health")
def health() -> Any:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/train")
def train():
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        # Invalidate cache so new model is picked up on next predict
        global _cached_model
        _cached_model = None
        model_path = _find_latest_model_path()
        return {"message": "Training completed", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: VisaBatchRequest):
    try:
        model = _load_model()
        # Convert list of pydantic models to DataFrame
        df = pd.DataFrame([item.dict() for item in req.items])
        # Model's preprocessing expects raw features; case_status is not included here
        # Predict
        preds = model.predict(df)
        # Map numeric outputs back to labels
        mapping = TargetValueMapping().reverse_mapping()
        labels = [mapping.get(int(p), str(p)) for p in preds]
        return {"predictions": labels}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))