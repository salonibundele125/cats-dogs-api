from fastapi import FastAPI
from app.predict import predict_image
from fastapi import FastAPI, File, UploadFile


app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = predict_image(contents)
    return {"prediction": prediction}