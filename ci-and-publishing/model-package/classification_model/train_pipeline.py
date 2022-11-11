from config.core import config
from imblearn.over_sampling import SMOTE
from pipeline import customer_pipe
from processing.data_manager import load_raw_dataset, save_model, save_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_raw_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        # independent variables
        data.drop(config.model_config.dropped_in_split, axis=1),
        # target variable
        data[config.model_config.target],
        # test size (0.2 in config)
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # fit model
    customer_pipe.fit(X_train, y_train)
    X_train = customer_pipe.transform(X_train)
    smote = SMOTE(sampling_strategy="minority", random_state=0)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=config.model_config.n_neighbors)
    knn_model.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=customer_pipe)
    save_model(model_to_persist=knn_model)


if __name__ == "__main__":
    run_training()
