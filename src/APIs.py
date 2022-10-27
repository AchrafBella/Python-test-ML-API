import pandas as pd
import uvicorn
import nest_asyncio
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from deserializer import check_json, deserialize_json
from model_training import clean_trainData, train_model, save_model, get_parameters
from applying_model import prepare_testData, apply_latestStatModel


class request_body(BaseModel):
    id: int    
    asset_scores: dict
    asset_infos: dict


class request_full_body(BaseModel):
    data_train_assets: list[request_body]


app = FastAPI(title='Deploying a ML Model with FastAPI')

@app.get("/")
def home():
    return "Welcome to this page that will allows you to use our functionalities"


@app.post("/check_data")
async def check_data(json_data: request_body) -> bool:
    """
    This function test if a json data respect the given correct format 
    :param json_data: one data structure 
    :return: boolean 
    """
    json_compatible_item_data = jsonable_encoder(json_data)
    if check_json(json_compatible_item_data):
        return True
    return False


@app.post("/model_information")
async def get_model_statistics(json_data: request_full_body) -> str:
    """
    This function take as input json file, structure the data, train the model and compute scores
    :param json_data: 
    :return: 
    """
    df = deserialize_json(json_data)
    # prepare the data
    x_train, x_test, y_train, y_test = clean_trainData(df, True, 0.20)
    # train the model 
    model = train_model(df, x_train, y_train, x_test, y_test, 100)
    # save the model 
    save_model(model)
    # compute scores 
    statistics = get_parameters(model, x_test, y_test)
    return statistics

    

@app.post("/model_predictions")
async def get_predictions(json_data: request_full_body) -> str:
    """
    This function take a json file and apply the latest model for predictions
    :param json_data: 
    :return: 
    """
    df = deserialize_json(json_data)
    X = prepare_testData(df)
    predictions = apply_latestStatModel(X)
    return {"predictions": str(predictions)}


nest_asyncio.apply()
uvicorn.run(app, port=6000)
