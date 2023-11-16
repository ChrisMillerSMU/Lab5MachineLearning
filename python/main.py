#!usr/bin/python
"""
NOTE:
    For this application to run properly, MongoDB must be running.


=========================================================
GLOBAL VARIABLES
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

# Standard library imports
from typing import List, Dict, Any, Union
from pprint import pprint

"""
=========================================================
GLOBAL VARIABLES
=========================================================
"""

# Initialize FastAPI app
app = FastAPI()

# MongoDB client setup with database name of `mydatabase` 
mongo_client: AsyncIOMotorClient = (
    AsyncIOMotorClient("mongodb://localhost:27017")
)
db = mongo_client.mydatabase

# clf is a global dictionary to store models
clf: Dict[int, Any] = {}

"""
=========================================================
PYDANTIC MODELS
=========================================================
"""

class PredictionRequest(BaseModel):
    feature: List[float] 
    dsid: int

class PredictionResponse(BaseModel):
    prediction: str

class DataPoint(BaseModel):
    feature: List[float]
    label: str
    dsid: int

class NewDatasetIDResponse(BaseModel):
    dsid: float

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
    Accepts a feature vector and a dataset ID (dsid), and uses the machine learning
    model associated with the dsid to make a prediction. If the model for the
    given dsid is not already loaded, it attempts to load it. If the model
    cannot be loaded or does not exist, an HTTPException is raised.

    Parameters
    ----------
    request : PredictionRequest
        A Pydantic model that includes a feature vector and a dataset ID (dsid).

    Returns
    -------
    PredictionResponse
        A Pydantic model that includes the prediction result as a string.
        
    Raises
    ------
    HTTPException
        An error response with status code 500 if the model cannot be loaded,
        or with status code 404 if no data can be found for the given dsid.

    Example
    -------
    POST /predict_one/
    {
        "feature": [0.1, 0.2, 0.3],
        "dsid": 123
    }
    Response:
    {
        "prediction": "Label1"
    }
    """
    fvals = get_features_as_SFrame(request.feature)
    dsid = request.dsid

    if dsid not in clf:
        return {"prediction": f"Model not calibrated yet for {dsid}"}

    # If the model is not loaded, attempt to load it
    if clf.get(dsid) is None:
        try:
            clf[dsid] = tc.load_model(f'../models/turi_model_dsid{dsid}')
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not load model for {dsid}: {str(e)}"
            )

    pred_label = clf[dsid].predict(fvals)
    return {"prediction": str(pred_label)}


@app.post("/upload_labeled_datapoint/")
async def upload_labeled_datapoint(data: DataPoint) -> Dict[str, Any]:
    """
    Receives a labeled data point and stores it in the database.
    The data point includes a feature vector, a label, and a dataset ID (dsid).

    Parameters
    ----------
    data: DataPoint
        The labeled data point to be stored, including its features, label, and dsid.

    Returns
    -------
    dict
        A dictionary containing the ID of the inserted data point and a summary of the features.
    """
    document: Dict[str, Any] = data.dict()

     # Ensure all features are float
    document["feature"] = [float(val) for val in document["feature"]] 
    result = await db.labeledinstances.insert_one(document)
    
    # Prepare the response object
    return {
        "id": str(result.inserted_id),
        "feature": [
            "{} Points Received".format(len(document["feature"])),
            "min of: " + str(min(document["feature"])),
            "max of: " + str(max(document["feature"])),
        ],
        "label": document["label"],
    }


@app.get("/get_new_dataset_id/", response_model=NewDatasetIDResponse)
async def get_new_dataset_id() -> NewDatasetIDResponse:
    """
    Retrieves a new, unique dataset ID that can be used for creating a new dataset.

    Returns
    -------
    NewDatasetIDResponse
        An object containing the new dataset ID.
    """
    latest = await db.labeledinstances.find_one(sort=[("dsid", -1)])
    new_session_id = 1 if latest is None else float(latest['dsid']) + 1
    return {"dsid": new_session_id}


@app.get("/update_model_for_dataset_id/")
async def update_model_for_dataset_id(dsid: int = 0):
    """
    Trains or updates a machine learning model for the specified dataset ID (dsid).
    If successful, saves the model and returns the resubstitution accuracy.

    Parameters
    ----------
    dsid: int
        The dataset ID for which the model should be trained or updated.

    Returns
    -------
    ResubAccuracyResponse
        An object containing the resubstitution accuracy of the trained or updated model.
    """
    features_labels = await get_features_and_labels_as_SFrame(dsid)
    if len(features_labels) > 0:
        model = tc.classifier.create(features_labels, target='target', verbose=0)
        yhat = model.predict(features_labels)
        clf[dsid] = model
        acc = sum(yhat == features_labels["target"]) / float(len(features_labels))
        model.save(f'../models/turi_model_dsid{dsid}')
        return {"resub_accuracy": acc}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="No data found for the given dsid."
        )


"""
=========================================================
HELPER FUNCTIONS
=========================================================
"""

def get_features_as_SFrame(vals: List[float]) -> tc.SFrame:
    """
    Converts a list of feature values into an SFrame that can be used for model prediction.

    Parameters
    ----------
    vals: List[float]
        A list of feature values.

    Returns
    -------
    SFrame
        An SFrame containing the feature values.
    """
    tmp = [float(val) for val in vals]
    tmp = np.array(tmp).reshape((1, -1))
    data = {'sequence': tmp}
    return tc.SFrame(data=data)


def get_features_and_labels_as_SFrame(dsid: int) -> tc.SFrame:
    """
    Retrieves feature vectors and labels from the database for a given dataset ID
    and converts them into an SFrame suitable for model training.

    Parameters
    ----------
    dsid: int
        The dataset ID for which features and labels should be retrieved.

    Returns
    -------
    SFrame
        An SFrame containing the features and labels.
    """
    # create feature vectors from database
    features: List[float] = []
    labels: List[str] = []
    for a in self.db.labeledinstances.find({"dsid": dsid}): 
        features.append([float(val) for val in a["feature"]])
        labels.append(a["label"])

    # convert to dictionary for tc
    data = {
        "target": labels, 
        "sequence": np.array(features),
    }

    # send back the SFrame of the data
    return tc.SFrame(data=data)


"""
=========================================================
MAIN METHOD
=========================================================
"""

if __name__ == "__main__":
    # Print existing route handlers
    print("Registered route handlers:")
    for route in app.routes:
        methods = ", ".join(route.methods)
        print(f"{methods} {route.path} -> {route.name}")
    
    # Start uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
