from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import googlenet

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
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {file_path}")
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = load_model('banana_freshness_model.pth')

# Function to make predictions
def predict_freshness(model, image):
    model.eval()
    with torch.no_grad():
        if isinstance(image, Image.Image):  # If it's a PIL image, transform it
            image = data_transforms(image).unsqueeze(0)  # Apply transforms and add batch dimension
        else:
            image = image.unsqueeze(0)  # Add batch dimension if it's already a tensor

        image = image.to(device)  # Move image to GPU or CPU
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

    if file:
        file_path = f"uploads/{file.filename}"
        file.save(file_path)
        image_path = file_path
        image = Image.open(image_path).convert('RGB')
        image = data_transforms(image)

        # Use the renamed function here
        freshness_index = predict_freshness(model, image)
        freshness_index = 100-(freshness_index - 1.8)/5.4*100

        if freshness_index<0: freshness_index = 0
        if freshness_index>100: freshness_index = 100

        print(freshness_index)
        return jsonify({"freshness_index": freshness_index})

if __name__ == "__main__":
    app.run(debug=True)
