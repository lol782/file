from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
import io

app = FastAPI()

# Load pre-trained model and processor
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Define waste classification mapping
biodegradable_items = {
    "apple": "Biodegradable", "banana": "Biodegradable", "paper": "Biodegradable",
    "leaves": "Biodegradable", "vegetables": "Biodegradable", "food": "Biodegradable",
    "cardboard": "Biodegradable", "cotton": "Biodegradable", "wood": "Biodegradable"
}
non_biodegradable_items = {
    "plastic": "Non-Biodegradable", "metal": "Non-Biodegradable", "glass": "Non-Biodegradable",
    "battery": "Non-Biodegradable", "electronics": "Non-Biodegradable", "bottle": "Non-Biodegradable",
    "can": "Non-Biodegradable", "aluminum": "Non-Biodegradable", "rubber": "Non-Biodegradable"
}

@app.post("/classify-waste/")
async def classify_waste(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Process image
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted label
    predicted_label = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_label].lower()

    # Assign category based on common relations
    category = "Unknown"
    reward = 0

    for item, cat in biodegradable_items.items():
        if item in predicted_class:
            category = cat
            reward = 10
            break

    for item, cat in non_biodegradable_items.items():
        if item in predicted_class:
            category = cat
            reward = -5
            break

    return {
        "filename": file.filename,
        "predicted_item": predicted_class,
        "category": category,
        "reward": reward
    }