# 🍌 Banana Freshness Checker

[![Flask](https://img.shields.io/badge/Built%20With-Flask-blue.svg)](https://flask.palletsprojects.com/) 
[![PyTorch](https://img.shields.io/badge/Built%20With-PyTorch-orange.svg)](https://pytorch.org/)
[![Render](https://img.shields.io/badge/Deployed%20On-Render-green.svg)](https://render.com/)

Check the freshness of bananas in a snap! This web app uses a Deep learning model to predict the freshness index of a banana based on its image. Simply upload a photo of a banana, and the app will analyze the image and return a freshness index from 0 to 100, where 100 represents maximum freshness.

### 🌐 [Try Here - banana-index.onrender.com](https://banana-index.onrender.com/)

## 📸 Features

- **Banana Freshness Prediction**: Upload an image of a banana, and the app will calculate a freshness score.
- **Simple, User-Friendly Interface**: Easily upload an image and get results with a single click.
- **Real-Time Processing**: Instantly get predictions without any delay.
- **Deployed on Render**: Robust and accessible from anywhere.

## 🖥️ Tech Stack

- **Flask**: Lightweight web framework for Python, used to build the app's backend.
- **PyTorch**: Used for the machine learning model and prediction processing.
- **Torchvision**: Provides access to pre-trained models and utilities for computer vision.
- **Render**: Cloud hosting platform used to deploy and host the app.


## 🚀 Getting Started

To run this project locally, follow these steps:

### Prerequisites

- Python 3.7 or higher
- pip package installer
  
### Installation

- Clone the repository:

  - git clone https://github.com/RatneshKJaiswal/Banana_Index
  - cd Banana_Index

- Install dependencies:

  - pip install -r requirements.txt

- Download the Model: Ensure that banana_freshness_model.pth (the pre-trained model) is in the project directory. If not available in this repo, download or add it manually.

- Run the App:

  - python app.py
  
- Access the App: Open your browser and go to http://127.0.0.1:5000 to use the app.

## 🌈 Usage

- Upload a Banana Image: Choose an image of a banana from your device.

- Get Freshness Index: Click "Check Freshness" to receive a score between 0 and 100, where:

  - 100: Fresh and ripe banana.
  - 0: Spoiled banana.
  
- Interpreting Results: The freshness index is calculated based on the appearance of the banana, such as color and texture.

## 📈 Model Details

The app uses a modified GoogLeNet model trained on images of bananas with varying freshness levels. The model analyzes input images and predicts a freshness score based on learned visual patterns.

- Input Size: The model processes images resized to 224x224 pixels.
- Normalization: Images are normalized using the mean and std values typical for pre-trained models.
  
### Detailed Project Description
The Banana Freshness Checker is designed to assist users in determining the ripeness of a banana based on visual appearance. Leveraging deep learning, the app employs a fine-tuned GoogLeNet model trained on banana images with varying stages of ripeness. Users can upload a banana image, and the model predicts a freshness score by analyzing color, texture, and other visual indicators. A score of 100 indicates a perfectly ripe banana, while 0 indicates spoilage. This tool provides an easy and interactive way for users to gauge banana freshness without any subjective interpretation.

## 🔧 Deployment
This app is deployed on Render. You can also deploy it on other platforms like Heroku or AWS by modifying the deployment settings.

- Deployment on Render

  - Push your code to a GitHub repository.
  - Link the GitHub repository to Render.

- Specify the start command:

  - python app.py

- Set the PORT environment variable in Render settings.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

- Fork the repository.
- Create a new branch.
- Commit your changes.
- Push to your branch and create a pull request.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📬 Contact

For questions or suggestions, please reach out to ratnesh.kr.jais@gmail.com.

Enjoy using the Banana Freshness Checker! 🍌
