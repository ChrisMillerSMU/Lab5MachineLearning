#!usr/bin/python
"""
NOTE:
    For this application to run properly, MongoDB must be running.


=========================================================
IMPORTS
=========================================================
"""

# Imports for managing server access, routing, and database logic
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

# Imports for handling data and ML model development & execution
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset
import torchaudio.transforms as T

# Standard library imports
import joblib  # To save and load Scikit-Learn models
import os
from typing import List, Tuple, Dict, Any

"""
=========================================================
GLOBAL VARIABLES, CLASSES, AND NECESSARY CODE BLOCKS TO EXECUTE
=========================================================
"""

# Initialize FastAPI app
app = FastAPI()

# MongoDB client setup with database name of `mydatabase` 
mongo_client: AsyncIOMotorClient = (
    AsyncIOMotorClient("mongodb://localhost:27017")
)
db = mongo_client.mydatabase

# Declare Logistic Regression model
logistic_model = LogisticRegression()

# Create a label encoder object
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse_output=False)

# Fit the label encoder and one-hot encoder with the known labels
known_labels = np.array(["Reece", "Ethan", "Rafe"])
label_encoder.fit(known_labels)
one_hot_encoder.fit(known_labels.reshape(-1, 1))

# Specify Mel Spectogram CNN Architecture
class MelSpectrogramCNN(nn.Module):
    def __init__(self):
        super(MelSpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Additional layers can be added here
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)  # Adjust the input dimensions based on your data
        self.fc2 = nn.Linear(1024, 3)  # Adjust the output dimensions based on the number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 32 * 32)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Declare Mel Spectogram model
spectogram_cnn = MelSpectrogramCNN()

# Function to load machine learning models from the file system.
def load_machine_learning_models():
    """
    Function to load machine learning models from the file system.
    """
    logistic_regression_path = "../ml_models/logistic_regression_model.pkl"
    mel_spectrogram_cnn_path = "../ml_models/mel_spectrogram_cnn.pth"

    # Load Logistic Regression model if exists, else create a new one
    if os.path.exists(logistic_regression_path):
        logistic_model = joblib.load(logistic_regression_path)
    else:
        logistic_model = LogisticRegression()
        joblib.dump(logistic_model, logistic_regression_path)

    # Load Mel Spectrogram CNN model if exists, else create a new one
    if os.path.exists(mel_spectrogram_cnn_path):
        spectogram_cnn = torch.load(mel_spectrogram_cnn_path)
    else:
        spectogram_cnn = MelSpectrogramCNN()
        torch.save(spectogram_cnn.state_dict(), mel_spectrogram_cnn_path)

    return {
        "Logistic Regression": logistic_model, 
        "Mel Spectogram CNN": spectogram_cnn, 
    }

# `model_dictionary` is a global dictionary to store machine learning models
model_dictionary: Dict[str, Any] = load_machine_learning_models() 

"""
=========================================================
PYDANTIC MODELS
=========================================================
"""

class PredictionRequest(BaseModel):
    raw_audio: List[float] 
    ml_model_type: str  # Model Types: "Logistic Regression", "Mel Spectogram CNN"

class PredictionResponse(BaseModel):
    audio_prediction: str  # Predictions: "Reece", "Ethan", "Rafe"

class DataPoint(BaseModel):
    raw_audio: List[float]
    audio_label: str  # "Reece", "Ethan", "Rafe"
    ml_model_type: str  # "Logistic Regression", "Mel Spectogram CNN"

class ResubAccuracyResponse(BaseModel):
    resub_accuracy: float

"""
=========================================================
ROUTES
=========================================================
"""

@app.post("/predict_one/", response_model=PredictionResponse)
async def predict_one(request: PredictionRequest) -> PredictionResponse:
    """
    Accepts a feature vector and a ML model type, and uses the machine learning
    model associated with the dsid to make a prediction. If the model for the
    given ML is not already loaded, it attempts to load it and make a prediction. 
    If the model cannot be loaded or does not exist, an HTTPException is raised.

    Parameters
    ----------
    request : PredictionRequest
        A Pydantic model that includes a feature vector and a model type.

    Returns
    -------
    PredictionResponse
        A Pydantic model that includes the prediction result as a string.
        
    Raises
    ------
    HTTPException
        An error response with status code 500 if the model cannot be loaded,
        or with status code 404 if no data can be found for the given model.

    Example
    -------
    POST /predict_one/
    {
        "feature": [0.0102, 0.2031, 0.923231, 0.0000123, ...],
        "model_type": "Logistic Regression"
    }
    Response:
    {
        "prediction": "Ethan"
    }
    """
    # Load in necessary variables and identify the model that will be used for the
    # prediction task
    feature_values = np.array(request.raw_audio)
    model_type: str = request.ml_model_type
    if model_dictionary.get(model_type) is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load model for {model_type}"
        )
    
    if model_type == "Logistic Regression":
        # Specify logistic regression model
        model = model_dictionary[model_type]

        # Predict using the feature values and reverse encode the prediction
        predicted_label_encoded = model.predict(feature_values.reshape(1,-1))
        audio_prediction = label_encoder.inverse_transform(predicted_label_encoded)
    
        # Return the predicted audio
        return {
            "audio_prediction": audio_prediction 
        }

    elif model_type == "Mel Spectogram CNN":
        # Convert raw audio data to Mel Spectrogram
        waveform = torch.tensor(request.raw_audio).float().view(1,-1)
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=44100, n_mels=128)
        mel_spectrogram = mel_spectrogram_transform(waveform)

        # Add a channel dimension and pass to the CNN
        mel_spectrogram = mel_spectrogram.view(1, 1, mel_spectrogram.size(1), mel_spectrogram.size(2))
        model = model_dictionary[model_type]

        # Predict using the Mel Spectrogram and reverse encode the prediction
        prediction = model(mel_spectrogram)
        predicted_label_index = prediction.argmax(dim=1)
        audio_prediction = known_labels[predicted_label_index.item()]

        # Return the predicted audio
        return {
            "audio_prediction": audio_prediction 
        }
    

@app.post("/upload_labeled_datapoint_and_update_model/")
async def upload_labeled_datapoint_and_update_model(data: DataPoint) -> Dict[str, Any]:
    """
    Receives a labeled data point and stores it in the database.
    The data point includes a feature vector, a label, and the model we'd like our
    data to be used in training. Then, the associated machine learning model for 
    the specified dataset ID is retrained. If successful, saves the model and returns 
    the resubstitution accuracy.

    Parameters
    ----------
    data: DataPoint
        The labeled data point to be stored, including its features, label, and model of interest

    Returns
    -------
    dict
        A dictionary containing the ID of the inserted data point and a summary of the features.
    """
    # Insert data into MongoDB
    insert_result = await db.labeledinstances.insert_one({
        "raw_audio": data.raw_audio,
        "audio_label": data.audio_label,
        "model_type": data.ml_model_type,
    })

    # Retrieve all data points for this model_type from MongoDB
    cursor = db.labeledinstances.find({"model_type": data.ml_model_type})
    data_points = await cursor.to_list(length=None)

    if data.ml_model_type == "Logistic Regression": 
        # Convert data to features and labels suitable for Logistic Regression
        features, labels = convert_to_numpy_dataset(data_points)

        # Train the model
        model, accuracy = retrain_logistic_regression_model(features, labels)

        # Update the model in the dictionary
        model_dictionary[data.ml_model_type] = model

        # Return the accuracy of the retrained model
        return {"resub_accuracy": accuracy}

    elif data.ml_model_type == "Mel Spectogram CNN":
        # Convert data to PyTorch dataset
        features, labels = convert_to_pytorch_dataset(data_points)

        # Train the model
        model, accuracy = await retrain_pytorch_model(features, labels)

        # Update the model in the dictionary
        model_dictionary[data.ml_model_type] = model

        # Return the accuracy of the trained model
        return {"resub_accuracy": accuracy}

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"No model found for {data.ml_model_type}"
        )

"""
=========================================================
HELPER FUNCTIONS
=========================================================
"""
def convert_to_numpy_dataset(data_points: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the list of data points to NumPy arrays for features and labels.
    """
    # Extract FFT features for Logistic Regression
    features_list = [np.fft.fft(np.array(dp["raw_audio"])).real for dp in data_points]
    labels_list = [dp["audio_label"] for dp in data_points]

    # Encode labels using label encoder
    labels_encoded = label_encoder.transform(labels_list)

    features = np.array(features_list)
    labels = labels_encoded

    return features, labels


def retrain_logistic_regression_model(
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[LogisticRegression, float]:
    """
    Retrain the Logistic Regression model using the provided features and labels.
    """
    model = model_dictionary["Logistic Regression"]
    model.fit(features, labels)

    # Evaluate accuracy
    accuracy = model.score(features, labels)
    return model, accuracy


def convert_to_pytorch_dataset(
    data_points: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert the list of data points to PyTorch Tensors for features and labels.
    """
    # Extract Mel Spectrogram features for CNN
    features_list = []
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=44100, n_mels=128)
    
    for dp in data_points:
        waveform = torch.tensor(dp['raw_audio']).float().view(1, -1)
        mel_spectrogram = mel_spectrogram_transform(waveform)
        mel_spectrogram = mel_spectrogram.view(1, mel_spectrogram.size(1), mel_spectrogram.size(2))
        features_list.append(mel_spectrogram)
    
    labels_list = [dp['audio_label'] for dp in data_points]

    # One-hot encode labels
    labels_encoded = one_hot_encoder.transform(np.array(labels_list).reshape(-1, 1))

    features = torch.stack(features_list)
    labels = torch.tensor(labels_encoded).float()

    return features, labels


async def retrain_pytorch_model(
    features: torch.Tensor, 
    labels: torch.Tensor,
) -> Tuple[nn.Module, float]:
    """
    Retrain the specified model using the provided features and labels.
    """
    model = model_dictionary["Mel Spectogram CNN"]

    # Define a simple dataset and dataloader
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):  # Train for 5 epochs
        for batch_features, batch_labels in dataloader:
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in dataloader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return model, accuracy


"""
=========================================================
MAIN METHOD
=========================================================
"""

if __name__ == "__main__":
    # Start uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
