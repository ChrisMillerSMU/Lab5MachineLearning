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
import json

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
    Use an existing machine learning model associated with the DSID
    to make a prediction based on the feature data passed in.

    Parameters
    ----------
    request: PredictionRequest
        dsid: int
            Dataset ID associated with machine learning model we use as our
            predictor.
        feature: List[float]
            Feature data that will be passed into our model.
    
    Returns
    -------
    PredictionResponse
        prediction: str
            The prediction label that the model predicts based on feature
            data.
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


@app.get("/print_handlers/")
async def print_handlers() -> str:
    """
    Prints out all of the routes for the viewer to see.
    Intended for debugging purposes and transparency for the user running
    this server.
    """
    routes_info: List[Dict[str, Union[str, List[str]]]] = [
        {"path": route.path, "name": route.name, "methods": list(route.methods)}
        for route in app.routes
    ]
    return json.dumps(routes_info, indent=2)


@app.post("/upload_labeled_datapoint/")
async def upload_labeled_datapoint(data: DataPoint) -> Dict[str, Any]:
    """
    
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


# Route to request a new dataset ID
@app.get("/get_new_dataset_id/", response_model=NewDatasetIDResponse)
async def get_new_dataset_id() -> NewDatasetIDResponse:
    latest = await db.labeledinstances.find_one(sort=[("dsid", -1)])
    new_session_id = 1 if latest is None else float(latest['dsid']) + 1
    return {"dsid": new_session_id}


# Route to update the model for a given dataset ID
@app.get("/update_model_for_dataset_id/")
async def update_model_for_dataset_id(dsid: int = 0):
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
    tmp = [float(val) for val in vals]
    tmp = np.array(tmp).reshape((1, -1))
    data = {'sequence': tmp}
    return tc.SFrame(data=data)


def get_features_and_labels_as_SFrame(dsid: int) -> tc.SFrame:
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
