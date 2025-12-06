from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from enum import Enum
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import requests
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enum for diagnosis
class DIAGNOSIS(str, Enum):
    NO_DEMENTED = "No Demente"
    VERY_MILD_DEMENTED = "Alzheimer Leve"
    MILD_DEMENTED = "Alzheimer Moderado"
    MODERATE_DEMENTED = "Alzheimer Severo"

# Request model
class ImageData(BaseModel):
    fileUrl: str
    fileName: str

class AnalysisRequest(BaseModel):
    patientId: str
    token: str
    images: List[ImageData]

# Load model and processor
model_name = "prithivMLmods/Alzheimer-Stage-Classifier"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)
# ID to label mapping
id2label = {
    "0": "MildDemented",
    "1": "ModerateDemented",
    "2": "NonDemented",
    "3": "VeryMildDemented"
}

def classify_alzheimer_stage(image):
    """
    Classify Alzheimer's stage from a PIL Image
    """
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    print(prediction)
    return prediction


def get_diagnosis_from_prediction(prediction: dict) -> DIAGNOSIS:
    """
    Get diagnosis based on highest percentage
    """
    # Find the key with highest value
    max_key = max(prediction, key=prediction.get)
    
    # Map model output to DIAGNOSIS enum
    diagnosis_mapping = {
        "NonDemented": DIAGNOSIS.NO_DEMENTED,
        "VeryMildDemented": DIAGNOSIS.VERY_MILD_DEMENTED,
        "MildDemented": DIAGNOSIS.MILD_DEMENTED,
        "ModerateDemented": DIAGNOSIS.MODERATE_DEMENTED
    }
    
    return diagnosis_mapping[max_key]


def download_image_from_url(url: str):
    """
    Download image from Cloudinary URL
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image


@app.get("/")
async def root():
    return {"message": "tamo activo papi"}


@app.post("/analyze")
async def analyze_images(request: AnalysisRequest):
    """
    Analyze multiple images from Cloudinary URLs and send results to NestJS backend
    """
    image_analyses = []
    
    # Process each image
    for image_data in request.images:
        try:
            # Download image from Cloudinary
            image = download_image_from_url(image_data.fileUrl)
            
            # Analyze image with AI model
            analysis = classify_alzheimer_stage(image)
            
            # Get diagnosis based on highest percentage
            diagnosis = get_diagnosis_from_prediction(analysis)
            
            # Convert percentages to 0-100 scale
            image_analysis = {
                "imageUrl": image_data.fileUrl,
                "fileName": image_data.fileName,
                "diagnosis": diagnosis.value,
                "nonDemented": round(analysis["NonDemented"] * 100, 2),
                "veryMildDemented": round(analysis["VeryMildDemented"] * 100, 2),
                "mildDemented": round(analysis["MildDemented"] * 100, 2),
                "moderateDemented": round(analysis["ModerateDemented"] * 100, 2)
            }
            
            image_analyses.append(image_analysis)
            
        except Exception as e:
            print(f"Error processing image {image_data.fileName}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing image {image_data.fileName}: {str(e)}"
            )
    
    # Prepare payload for NestJS backend
    payload = {
        "patientId": request.patientId,
        "imageAnalysis": image_analyses
    }
    
    # Send results to NestJS backend
    try:
        nestjs_backend_url = os.getenv("NESTJS_BACKEND_URL", "http://localhost:3000/api/v1/analysis")
        nestjs_response = requests.post(
            nestjs_backend_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {request.token}"
            },
            timeout=30
        )
        nestjs_response.raise_for_status()
        
        # Return the response from NestJS backend
        return nestjs_response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to NestJS backend: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error enviando datos al backend: {str(e)}"
        )