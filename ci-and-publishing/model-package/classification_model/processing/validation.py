from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter. This function
    is mostly unneeded for this project considering missing values
    are already handled inside the pipeline."""

    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if validated_data[var].isnull().sum() > 0
    ]
    validated_data = validated_data.dropna(subset=new_vars_with_na)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values. This function is
    also unneeded because missing values are processed inside
    the pipeline."""

    relevant_data = input_data.copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleCustomerDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class CustomerDataInputSchema(BaseModel):
    ID: Optional[int]
    Year_Birth: Optional[int]
    Education: Optional[str]
    Marital_Status: Optional[str]
    Income: Optional[float]
    Kidhome: Optional[int]
    Teenhome: Optional[int]
    Dt_Customer: Optional[str]
    Recency: Optional[int]
    MntWines: Optional[int]
    MntFruits: Optional[int]
    MntMeatProducts: Optional[int]
    MntFishProducts: Optional[int]
    MntSweetProducts: Optional[int]
    MntGoldProds: Optional[int]
    NumDealsPurchases: Optional[int]
    NumWebPurchases: Optional[int]
    NumCatalogPurchases: Optional[int]
    NumStorePurchases: Optional[int]
    NumWebVisitsMonth: Optional[int]
    AcceptedCmp3: Optional[int]
    AcceptedCmp4: Optional[int]
    AcceptedCmp5: Optional[int]
    AcceptedCmp1: Optional[int]
    AcceptedCmp2: Optional[int]
    Complain: Optional[int]
    Z_CostContact: Optional[int]
    Z_Revenue: Optional[int]


class MultipleCustomerDataInputs(BaseModel):
    inputs: List[CustomerDataInputSchema]
