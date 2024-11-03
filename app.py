import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import googlenet
import io

app = Flask(__name__)

# Define the model architecture (same as used during training)
model = googlenet(weights=None, aux_logits=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)

# Define image transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(file_path='banana_freshness_model.pth'):
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {file_path}")
    return model


# Set device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = load_model('banana_freshness_model.pth')


# Function to make predictions
def predict_freshness(model, image):
    model.eval()
    with torch.no_grad():
        image = data_transforms(image).unsqueeze(0).to(device)
        output = model(image)
        return output.item()


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for processing the image upload and returning the freshness index
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Process the uploaded image from memory
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    freshness_score = predict_freshness(model, image)

    # Scale the freshness score to a 0-100 range
    freshness_index = 100 - ((freshness_score - 1.8) / 5.4) * 100
    freshness_index = max(0, min(freshness_index, 100))  # Clip between 0 and 100

    return jsonify({"freshness_index": round(freshness_index, 2)})


# Start the Flask app using the specified PORT environment variable
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)