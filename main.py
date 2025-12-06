from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import requests
from io import BytesIO


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ImageData(BaseModel):
    fileUrl: str
    fileName: str

class AnalysisRequest(BaseModel):
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
    Analyze multiple images from Cloudinary URLs
    Returns array with url, fileName and analysis for each image
    """
    results = []
    
    for image_data in request.images:
        try:
            # Download image from Cloudinary
            image = download_image_from_url(image_data.fileUrl)
            
            # Analyze image with AI model
            analysis = classify_alzheimer_stage(image)
            
            # Append result
            results.append({
                "fileUrl": image_data.fileUrl,
                "fileName": image_data.fileName,
                "analysis": analysis
            })
            
        except Exception as e:
            # If error occurs, include error message
            results.append({
                "fileUrl": image_data.fileUrl,
                "fileName": image_data.fileName,
                "error": str(e)
            })
    
    return {"results": results}