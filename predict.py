import bentoml
from pydantic import BaseModel
from bentoml.io import JSON, NumpyNdarray
import numpy as np
import pandas as pd

class IncomeInput(BaseModel):
    Age: int
    Workclass: str
    Education: str 
    MaritalStatus: str
    Occupation: str
    Relationship: str
    Race: str 
    Sex: str 
    CapitalGain: int 
    CapitalLoss: int
    HoursPerWeek: int 
    NativeCountry: str

model = bentoml.sklearn.get("adult_xgboost:latest")
model_runner = model.to_runner()
preprocessor = model.custom_objects['preprocessor']

service = bentoml.Service("adult_xgboost_service", runners=[model_runner])

input_spec = JSON(pydantic_model=IncomeInput)

@service.api(input=input_spec, output=NumpyNdarray())
async def predict(input_data: IncomeInput) -> np.ndarray:
    X = pd.DataFrame([input_data.dict()])
    X_model = preprocessor.transform(X)
    output = await model_runner.predict.async_run(X_model)
    return output
