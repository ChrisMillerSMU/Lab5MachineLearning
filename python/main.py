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
from fastapi import FastAPI, HTTPException, Body, status
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

# Imports for handling data and ML model development & execution
import turicreate as tc
import numpy as np

# Machine learning packages
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchaudio.transforms as T

# Standard library imports
from typing import List, Tuple, Dict, Any, Union

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

# Create modes for machine learning
model_types = ["Logistic Regression", "Mel's Spectogram CNN"]

# Declare Logistic Regression model
logistic_model = LogisticRegression()

# Specify Mel's Spectogram CNN Architecture
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

# `model_dictionary` is a global dictionary to store models
model_dictionary: Dict[str, Any] = {
    "Logistic Regression": logistic_model, 
    "Mel's Spectogram CNN": spectogram_cnn, 
}

"""
=========================================================
PYDANTIC MODELS
=========================================================
"""

class PredictionRequest(BaseModel):
    raw_audio: List[float] 
    model_type: str  # Model Types: "Spectogram CNN", "Mel's Spectogram CNN"

class PredictionResponse(BaseModel):
    audio_prediction: str  # Predictions: "Reece", "Ethan", "Rafe"

class DataPoint(BaseModel):
    raw_audio: List[float]
    audio_label: str  # "Reece", "Ethan", "Rafe"
    model_type: str  # "Spectogram CNN", "Mel's Spectogram CNN"

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
        "feature": [0.1, 0.2, 0.3, ...],
        "model_type": 123
    }
    Response:
    {
        "prediction": "Ethan"
    }
    """
    # Load in necessary variables and identify the model that will be used for the
    # prediction task
    feature_values = np.array(request.raw_audio)
    model_type: str = request.model_type
    if model_dictionary.get(model_type) is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load model for {model_type}"
        )
    
    if model_type == "Logistic Regression":
        # Specify logistic regression model
        model = model_dictionary[model_type]

        # Prediict using the feature values
        return {
            "audio_prediction": model.predict(feature_values.reshape(1,-1)) 
        }
    elif model_type == "Mel's Spectogram CNN":
        # Convert raw audio data to Mel Spectrogram
        waveform = torch.tensor(request.raw_audio).float().view(1, -1)
        mel_spectrogram_transform = T.MelSpectrogram(sample_rate=22050, n_mels=128)
        mel_spectrogram = mel_spectrogram_transform(waveform)

        # Add a channel dimension and pass to the CNN
        mel_spectrogram = mel_spectrogram.view(1, 1, mel_spectrogram.size(1), mel_spectrogram.size(2))
        model = model_dictionary[model_type]

        # Predict using the Mel Spectrogram
        prediction = model(mel_spectrogram)
        predicted_label = prediction.argmax(dim=1)

        # Map the label index to actual label name, assuming labels are stored in a list
        labels = ["Reece", "Ethan", "Rafe"]
        return {
            "audio_prediction": labels[predicted_label.item()]
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
        "model_type": data.model_type,
    })

    # Retrieve all data points for this model_type from MongoDB
    cursor = db.labeledinstances.find({"model_type": data.model_type})
    data_points = await cursor.to_list(length=None)

    if data.model_type == "Logistic Regression": 
        # Convert data to features and labels suitable for Logistic Regression
        features, labels = convert_to_numpy_dataset(data_points)

        # Train the model
        accuracy = retrain_logistic_regression_model(features, labels)

        # Return the accuracy of the retrained model
        return {"resub_accuracy": accuracy}

    elif data.model_type == "Mel's Spectogram CNN":
        # Convert data to PyTorch dataset
        features, labels = convert_to_pytorch_dataset(data_points)

        # Train the model
        model, accuracy = await retrain_pytorch_model(features, labels)

        # Update the model in the dictionary
        model_dictionary[data.model_type] = model
        return {"resub_accuracy": accuracy}

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"No model found for {data.model_type}"
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
    features_list = [np.fft.fft(np.array(dp['raw_audio'])).real for dp in data_points]
    labels_list = [dp['audio_label'] for dp in data_points]

    # Convert features and labels list to NumPy array
    features = np.array(features_list)
    labels = np.array(labels_list)

    return features, labels


def retrain_logistic_regression_model(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Retrain the Logistic Regression model using the provided features and labels.
    """
    model = model_dictionary["Logistic Regression"]
    model.fit(features, labels)

    # Evaluate accuracy
    accuracy = model.score(features, labels)
    return accuracy


async def convert_to_pytorch_dataset(data_points: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert the list of data points to PyTorch Tensors for features and labels.
    """
    # Extract Mel Spectrogram features for CNN
    features_list = []
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=22050, n_mels=128)
    
    for dp in data_points:
        waveform = torch.tensor(dp['raw_audio']).float().view(1, -1)
        mel_spectrogram = mel_spectrogram_transform(waveform)
        mel_spectrogram = mel_spectrogram.view(1, mel_spectrogram.size(1), mel_spectrogram.size(2))
        features_list.append(mel_spectrogram)
    
    labels_list = [dp['audio_label'] for dp in data_points]

    features = torch.stack(features_list)
    labels = torch.tensor(labels_list).long()  # Assuming labels are encoded as integers

    return features, labels


async def retrain_pytorch_model(
    features: torch.Tensor, 
    labels: torch.Tensor,
) -> Tuple[nn.Module, float]:
    """
    Retrain the specified model using the provided features and labels.
    """
    model = model_dictionary["Mel's Spectogram CNN"]

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
