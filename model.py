import glob
import joblib
import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings
from typing import List
warnings.filterwarnings("ignore")

seed = 42

class MachineLearningModel:
    def __init__(self, features: List[str]):
        self.model = None
        self.features = features

        self.model_mapping = {
            'random_forest': RandomForestRegressor(random_state=seed),
            'linear_regression': LinearRegression(),
            'svr': SVR(),
            'decision_tree': DecisionTreeRegressor(random_state=seed),
            'gradient_boosting': GradientBoostingRegressor(random_state=seed)
        }

    def load_data(self, data_path: str = 'data') -> None:
        data_files = glob.glob(f'{data_path}/*.csv')
        return pd.concat((pd.read_csv(file) for file in data_files), ignore_index=True)

    def train(self, model_name: str, data_path: str = 'data'):
        data = self.load_data(data_path)
        model_names = [model_name] if model_name else self.model_mapping.keys()
        best_estimators = {}
        parameter_distributions = {
            RandomForestRegressor : dict(
                n_estimators =[10, 100],
                min_samples_leaf = [1, 2, 4, 8, 16],
                min_samples_split = [2, 4, 8, 16]
            ),
            SVR: dict(
                C =[0.1, 1, 10],
                gamma=['scale', 'auto'],
                kernel=['linear', 'rbf']
            )
        }
        for name in model_names:
            model_class = self.model_mapping[name]

            grid = RandomizedSearchCV(
                estimator = model_class,
                param_distributions = parameter_distributions.get(type(model_class), {}),
                cv=5,
                n_iter=20,
                random_state=seed
            )
            grid.fit(data[self.features], data['y'])
            joblib.dump(grid.best_estimator_, f"models/{name}_model.joblib")
            best_estimators[name] = grid.best_estimator_
        self.model = best_estimators

    def predict(self, datapoint: List[str], model_name: str) -> List[float]:
        try:
            self.model = joblib.load(f"models/{model_name}_model.joblib")
            print(f"loaded {model_name} from disk")
            features = pd.DataFrame(datapoint, columns=self.features)
            return self.model.predict(features).tolist()
        except Exception as e:
            print(f"Error: {e}")