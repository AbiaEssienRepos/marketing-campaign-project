import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DropConstant:
    """Drops features with constant values"""

    def __init__(self):
        return None

    def fit(self, X, y=None):

        # learn and persist the variables with constant values
        self.constant_ = [feature for feature in X.columns if X[feature].nunique() == 1]
        return self

    def transform(self, X, y=None):

        X = X.drop(self.constant_, axis=1)
        return X


class MissingImputer:
    """Imputes the mean in place of missing or null values"""

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

        self.params_ = {}
        for variable in self.variables:
            self.params_[variable] = {}

    def fit(self, X, y=None):

        for variable in self.variables:
            self.params_[variable] = X[variable].mean()

        return self

    def transform(self, X, y=None):

        for variable in self.variables:
            X[variable] = np.where(
                X[variable].isnull(), self.params_[variable], X[variable]
            )

        return X


class TransformDate:
    """Converts date of first purchase to patronage period"""

    def __init__(self, variables, current_year):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def adjust_date(self, row):

        year = int(row.split("-")[0])
        patronage_period = self.current_year - year
        return patronage_period

    def transform(self, X, y=None):

        for variable in self.variables:
            X[variable] = X[variable].apply(self.adjust_date)

        return X


class TransformYear:
    """Converts date of birth to customer's age"""

    def __init__(self, variables, current_year):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.current_year = current_year

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        for variable in self.variables:
            X[variable] = self.current_year - X[variable]

        return X


class OrdinalEncoder:
    """Performs ordinal encoding on non-binary variables"""

    def __init__(self, variables, target):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.target = target

        # to store the parameters from the training data
        # the parameters are the YES rates for each of the labels in the train set
        self.params_ = {}
        for variable in self.variables:
            self.params_[variable] = {}

        # to store the ordinal rankings learned from the YES proportions
        self.ordinal_labels_ = {}

    def fit(self, X, y):

        X = pd.concat([X, y], axis=1)

        for variable in self.variables:

            # iterate over the labels
            for label in X[variable].unique():

                # grab all the rows for each label where the target = YES
                label_yes = len(X[(X[variable] == label) & (X[self.target] == 1)])
                label_size = len(X[X[variable] == label])

                # persist the YES proportion to its respective label in the dictionary
                self.params_[variable][label] = label_yes / label_size

            # rank the labels per YES proportions
            # labels with high YES rates get the highest rankings
            labels = pd.Series(self.params_[variable])
            ordered_labels = labels.sort_values().index
            ordinal_label = {k: i for i, k in enumerate(ordered_labels, 1)}

            # persist the rankings to the ordinal labels dictionary
            self.ordinal_labels_[variable] = ordinal_label

        return self

    def transform(self, X, y=None):

        for variable in self.variables:
            X[variable] = X[variable].map(self.ordinal_labels_[variable])

        return X


class ContinuousScaler:
    """Scales and returns a chosen subset of continuous variables"""

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X, y=None):
        # learn and persist the mean and standard deviation
        # of the dataset

        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(X[self.variables])
        return self

    def transform(self, X, y=None):

        X[self.variables] = self.scaler_.transform(X[self.variables])
        return X
