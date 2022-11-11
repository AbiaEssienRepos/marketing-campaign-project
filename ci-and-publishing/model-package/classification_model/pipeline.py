# from scikit-learn
from sklearn.pipeline import Pipeline

from classification_model.config.core import config

# our in-house pre-processing module
from classification_model.processing import preprocessors as pp

# set up the pipeline
customer_pipe = Pipeline(
    [
        # ===== CONSTANT VALUES =====
        # drop variables with constant values from the dataset
        ("drop_constants", pp.DropConstant()),
        # ===== MEAN IMPUTATION =====
        # replace null values with the variable mean
        (
            "mean_imputation",
            pp.MissingImputer(variables=config.model_config.missing_vals),
        ),
        # ===== TEMPORAL VARIABLES =====
        # ===== Dt_Customer =====
        (
            "transform_date",
            pp.TransformDate(
                variables=config.model_config.date_var,
                current_year=config.model_config.current_year,
            ),
        ),
        # ===== Year_Birth =====
        (
            "transform_year",
            pp.TransformYear(
                variables=config.model_config.year_var,
                current_year=config.model_config.current_year,
            ),
        ),
        # ===== ENCODING =====
        # encode non-binary variables
        (
            "non_binary_encoder",
            pp.OrdinalEncoder(
                variables=config.model_config.non_binary,
                target=config.model_config.target,
            ),
        ),
        # ===== SCALER =====
        # scale the continuous variables
        ("scaler", pp.ContinuousScaler(variables=config.model_config.scaled_vars)),
    ]
)
