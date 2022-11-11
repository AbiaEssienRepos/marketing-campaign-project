import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import (
    DATASET_DIR,
    TRAINED_KNN_DIR,
    TRAINED_MODEL_DIR,
    config,
)


def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe["MSSubClass"] = dataframe["MSSubClass"].astype("O")

    # rename variables beginning with numbers to avoid syntax errors later
    transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the transformer pipeline.
    Saves the versioned transformer, and overwrites any previous
    saved transformer. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def save_model(*, model_to_persist: KNeighborsClassifier) -> None:
    """Persist the model.
    Saves the versioned knn model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    save_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
    save_path = TRAINED_KNN_DIR / save_file_name

    do_not_delete = ["__init__.py", save_file_name]
    for model_file in TRAINED_KNN_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
    joblib.dump(model_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def load_model(*, file_name: str) -> KNeighborsClassifier:
    """Load a persisted model."""

    file_path = TRAINED_KNN_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
