from fastapi import FastAPI
#import uvicorn[standard]
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
from enum import  Enum
from typing import Union
from pydantic import BaseModel
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler



from typing import List

app = FastAPI()
# load pretrained data
model = pickle.load(open('./dumps/model.pkl','rb'))
onehoter = pickle.load(open('./dumps/one.pkl', 'rb'))
scaler = pickle.load(open('./dumps/skal.pkl', 'rb'))


model_numeric_01= [ 'year', 'km_driven', 'mileage',  'engine',  'max_power', 'seats']
model_categiric_01= [  'fuel',  'seller_type',  'transmission',  'owner']


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def data_preparator(raw_data):
    '''
    prepare data - scaling and normalising
    :param raw_data:
    :return:
    '''
    # numeric data
    data_numeric = raw_data[model_numeric_01]
    data_numeric = scaler.transform(data_numeric)
    # categorical data
    data_categirical = raw_data[model_categiric_01]
    data_categirical = pd.DataFrame(onehoter.transform(data_categirical))
    # join all type
    data = pd.DataFrame(np.hstack([data_numeric, data_categirical]))
    data.reindex
    return data


def data_cleaner(raw_data):
    '''
    clean data - work with missings, dublicates, new values, strings and simbols prepare and other
    :param raw_data: real data
    :return: data for model use
    '''

    # Но в задании этого не требовалось, а времени как всегда не хватает.
    # Предположим, что Заказчика предупредили о новом формате данных или выдали ему валидатор
    # Иначе тут до бесконечности можно все совершенствовать....
    return raw_data

def get_predicted(val):
    data = pd.DataFrame(jsonable_encoder(val))
    pred_train = model.predict(data_preparator(data)) # predict
    return list(pred_train)



@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return  get_predicted([item])[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return get_predicted(items)