from pydantic import BaseModel,Field
from typing import List, Literal,Optional



class CustomerInput(BaseModel):

    # Numerical Features
    customer_id: Optional[str]=Field(default=None)
    tenure: Optional[int]=Field(default=None,ge=0)
    MonthlyCharges:Optional[float]=Field(default=None,ge=0)
    TotalCharges:Optional[float]=Field(default=None,ge=0)

    #Categorical Features
    gender:Optional[Literal["Male","Female"]]=Field(default=None)
    Dependents:Optional[Literal["Yes","No"]]=Field(default=None)
    Contract:Optional[Literal["Month-to-month","One year","Two year"]]=Field(default=None)


class ChurnReason(BaseModel):
    feature:str
    impact:float
    direction:str


class PredictionOutput(BaseModel):
    customer_id:str
    churn_probability:float=Field(...,ge=0,le=1.0)
    churn_prediction:Literal[0,1]
    risk_tier:Literal["Low","Medium","High"]
    top_reasons:List[ChurnReason]
    model_version:str
    timestamp:str


class RowWarning(BaseModel):
    row_number:int
    customer_id:Optional[str]=None
    issue:str


class BatchPredictionResponse(BaseModel):
    total_rows_receieved:int
    total_processed:int
    skipped_count:int
    high_risk_count:int
    predictions:List[PredictionOutput]
    warnings:List[RowWarning]

class HealthResponse(BaseModel):
    status:str
    model_loaded:bool
    model_version:str


