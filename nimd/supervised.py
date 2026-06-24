import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    cross_val_score,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    train_test_split,
)
from sklearn.metrics import (
    f1_score,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)
from typing import Literal, List

from .core.utils import Score


# TODO: 1) CV implementation, 2) ft the models, 3) create def evaluates for Xs and Ws, 4) scores types


class SupervisedLearning:
    def __init__(
        self,
        task: Literal["classification", "regression"],
        val_train_test_splits: List,
        modelname: Literal["rf", "linreg", "logit", "mlp"],
        cv: bool,
        rng: int = 42,
        get_shap: bool = False,
    ):
        self.task = task
        self.cv = cv
        # self.score_type = score_type
        # TODO: I need a validation set for tuning of supervised models
        self.val_train_test_splits = val_train_test_splits
        self.rng = rng
        self.modelname = modelname
        self.get_shap = get_shap

        # null fields - use as getter
        self.model: RandomForestRegressor = None
        self.shap_values: shap.Explanation = None
        self.shap_explainer = None

    def balanced_sample(self, df: pd.DataFrame, label_col, n_per_class):
        groups = []
        for label, group in df.groupby(label_col):
            groups.append(group.sample(n=n_per_class, replace=len(group) < n_per_class))
        return pd.concat(groups).sample(frac=1).reset_index(drop=True)

    def get_model(self):
        return self.model

    def get_shap_explainer(self):
        return self.shap_explainer

    def get_shap_values(self):
        """Global feature importance with average impact on model output magnitude."""
        if self.get_shap:
            vals = np.abs(self.shap_values.values).mean(0)
        else:
            vals = None
        return vals

    def supervised_learning(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ):
        if self.task == "regression":
            if self.modelname == "linreg":
                model = LinearRegression(n_jobs=-1)
            if self.modelname == "rf":
                model = RandomForestRegressor(
                    n_estimators=100, random_state=self.rng, n_jobs=-1
                )
            if self.modelname == "mlp":
                model = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=self.rng,
                )
            scoring = "r2"
            cv_strategy = RepeatedKFold(n_splits=5, n_repeats=10, random_state=self.rng)
        elif self.task == "classification":
            model = LogisticRegression(n_jobs=-1)
            scoring = "accuracy"
            cv_strategy = RepeatedStratifiedKFold(
                n_splits=5, n_repeats=10, random_state=self.rng
            )
        if self.cv == False:
            model = model.fit(X_train, y_train)
            if self.get_shap:
                self.shap_explainer = shap.Explainer(model)
                self.shap_values = self.shap_explainer(X_test)
            y_pred = model.predict(X_test)
            self.model = model
        elif self.cv == True:
            # TODO: ensure CV has the different scores from self.score_type
            X = np.concatenate((X_train, X_test), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)
            self.model = model
        scores = Score(
            true=y_test,
            pred=y_pred,
            test_size=X_test.shape[0],
            train_size=X_train.shape[0],
            r2=r2_score(y_true=y_test, y_pred=y_pred),
            accuracy=accuracy_score(
                y_true=y_test.astype(int), y_pred=y_pred.astype(int)
            ),
            f1=f1_score(
                y_true=y_test.astype(int),
                y_pred=y_pred.astype(int),
                average=None,
            ),
            f1_macro=f1_score(
                y_true=y_test.astype(int),
                y_pred=y_pred.astype(int),
                average="macro",
            ),
            mae=mean_absolute_error(y_true=y_test, y_pred=y_pred),
            rmse=root_mean_squared_error(y_true=y_test, y_pred=y_pred),
        )
        return scores

    def evaluate(
        self, X, W, y, evaluation_type: Literal["full", "feature", "balanced"], **kwargs
    ):
        """
        Select 'full' vs 'feature' matrix evaluation and pass to supervised_learning()
        """
        if evaluation_type == "full" and X is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.val_train_test_splits[2],
                train_size=self.val_train_test_splits[1],
                random_state=self.rng,
            )
        elif evaluation_type == "balanced":
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.val_train_test_splits[2],
                train_size=self.val_train_test_splits[1],
                random_state=self.rng,
            )
            train_df = pd.concat([X_train, y_train], axis=1)
            train_bal = self.balanced_sample(
                train_df, label_col="label", n_per_class=y_train.value_counts().min()
            )
            X_train = train_bal.drop(columns=["label"])
            y_train = train_bal["label"]
        elif evaluation_type == "feature" and W is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                W,
                y,
                test_size=self.val_train_test_splits[2],
                train_size=self.val_train_test_splits[1],
                random_state=self.rng,
            )
        elif X is None and W is None:
            X_train = kwargs.get("X_train")
            X_test = kwargs.get("X_test")
            y_train = kwargs.get("y_train")
            y_test = kwargs.get("y_test")

        return self.supervised_learning(X_train, X_test, y_train, y_test)
