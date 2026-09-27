from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI(
    title="Email Spam Classifier API",
    description="API for classifying emails as spam or ham",
    version="1.0.0"
)


# Load trained model
model = joblib.load("models/spam_classifier_svm.joblib")


class EmailRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Email Spam Classifier API is running"
    }


@app.post("/predict")
def predict_email(request: EmailRequest):

    prediction = model.predict([request.text])[0]

    return {
        "prediction": prediction
    }