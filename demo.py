from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from Visa.pipline.training_pipeline import TrainingPipeline


pipeline = TrainingPipeline()
pipeline.run_pipeline()