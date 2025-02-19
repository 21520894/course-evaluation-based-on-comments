from fastapi import FastAPI
import joblib
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import numpy as np
import azure.cosmos.documents as documents
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey

from fastapi.responses import ORJSONResponse
import json

import config

HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']
CONTAINER_ID = config.settings['container_id']

app = FastAPI()

client = cosmos_client.CosmosClient(HOST, {'masterKey': MASTER_KEY})
database = client.create_database_if_not_exists(id=DATABASE_ID)
print('Database with id \'{0}\' was found'.format(DATABASE_ID))
container = database.create_container_if_not_exists(
    id=CONTAINER_ID,
    partition_key=PartitionKey(path='/partitionKey'),
    offer_throughput=400
)


model = joblib.load('./course_clf.pkl')

class Course(BaseModel):
    id: str
    name: str
    resource: int
    total_comments: int
    average_completion_rate: float
    negative: int
    positive: int
    neutral: int
    num_users: int
    course_classification: str | None = None

@app.get("/")
def hello():
    return {'res':'hello'}

@app.get("/getdata")
def getData():
    item_list = container.read_all_items()
    item_list = list(item_list)

    json_data = json.dumps(item_list,ensure_ascii=False)
    return ORJSONResponse(json_data)


@app.post("/predict")
def predict(course: Course):
    course = course.dict()
    course.pop('id', None)
    course.pop('name', None)
    course.pop('course_classification', None)
    if course['num_users']< 30: 
        return {"res": "NOT ENOUGH USER"}
    else: 
        X = list(course.values())
        print(X)
        pred = model.predict([X])
        return {"res": str(pred[0])}

# if __name__ == "__main__":
#     app