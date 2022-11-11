from typing import Any, List, Optional

from classification_model.processing.validation import CustomerDataInputSchema
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleCustomerDataInputs(BaseModel):
    inputs: List[CustomerDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Year_Birth": 1972,
                        "Education": "Graduation",
                        "Marital_Status": "Married",
                        "Income": 65685.0,
                        "Kidhome": 0,
                        "Teenhome": 1,
                        "Dt_Customer": "2014-03-29",
                        "Recency": 54,
                        "MntWines": 642,
                        "MntFruits": 14,
                        "MntMeatProducts": 49,
                        "MntFishProducts": 0,
                        "MntSweetProducts": 7,
                        "MntGoldProds": 57,
                        "NumDealsPurchases": 1,
                        "NumWebPurchases": 9,
                        "NumCatalogPurchases": 2,
                        "NumStorePurchases": 9,
                        "NumWebVisitsMonth": 5,
                        "AcceptedCmp3": 0,
                        "AcceptedCmp4": 0,
                        "AcceptedCmp5": 0,
                        "AcceptedCmp1": 0,
                        "AcceptedCmp2": 0,
                        "Complain": 0,
                        "Z_CostContact": 3,
                        "Z_Revenue": 11,
                    }
                ]
            }
        }
