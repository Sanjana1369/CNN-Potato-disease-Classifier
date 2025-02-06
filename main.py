from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your pre-trained model
MODEL = tf.keras.models.load_model("./saved_models/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Serve the HTML form at the root URL
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <title>Potato Plant Disease Prediction</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #f0f0f0;
                    margin: 0;
                }
                .container {
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }
                h1 {
                    color: #4CAF50;
                }
                form {
                    margin-top: 20px;
                }
                input[type="file"] {
                    margin-bottom: 20px;
                }
                input[type="submit"] {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
                #loading {
                    display: none;
                    font-size: 18px;
                    color: #4CAF50;
                    margin-top: 20px;
                }
                #image-preview {
                    max-width: 300px;
                    max-height: 300px;
                    margin-bottom: 20px;
                }
                #result {
                    font-size: 18px;
                    margin-top: 20px;
                    color: #333;
                }
            </style>
            <script>
                function showLoading() {
                    document.getElementById('loading').style.display = 'block';
                }

                function previewImage() {
                    const file = document.getElementById('file-input').files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            document.getElementById('image-preview').src = e.target.result;
                        };
                        reader.readAsDataURL(file);
                    }
                }

                function showResult(result) {
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = `<strong>Class:</strong> ${result.class}<br><strong>Confidence:</strong> ${result.confidence.toFixed(2)}`;
                    document.getElementById('loading').style.display = 'none';
                }

                async function handleFormSubmit(event) {
                    event.preventDefault();
                    const fileInput = document.getElementById('file-input');
                    if (!fileInput.files.length) {
                        alert("Please select an image file before submitting!");
                        return;
                    }

                    showLoading();

                    const formData = new FormData();
                    formData.append("file", fileInput.files[0]);

                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    showResult(result);
                }
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Potato Plant Disease Prediction</h1>
                <form id="upload-form" enctype="multipart/form-data" onsubmit="handleFormSubmit(event)">
                    <img id="image-preview" src="#" alt="Image Preview" />
                    <input id="file-input" name="file" type="file" accept="image/*" onchange="previewImage()" />
                    <input type="submit" value="Submit">
                </form>
                <div id="loading">Processing your image...</div>
                <div id="result"></div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=content)

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image = image.resize((224, 224))  # Resize the image to match the input size expected by your model
        return np.array(image)
    except Exception as e:
        print(f"Error reading the image: {e}")
        return None

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    if image is None:
        return {"error": "Could not process the image. Ensure it is a valid image format."}

    img_batch = np.expand_dims(image, 0)  # Expand dimensions to add batch size

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
