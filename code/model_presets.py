from typing import Dict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator

def defaultEstimatorModels() -> Dict[str, BaseEstimator]:
    linear_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'BayesianRidge': BayesianRidge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'LassoLars': LassoLars(),
        'SGDRegressor': SGDRegressor()
    }

    svr_models = {
        'LinearSVR': LinearSVR(),
        'NuSVR': NuSVR(),
        'SVR': SVR()
    }

    ensemble_models = {
        'Random Forest': RandomForestRegressor()
    }

    neighbors_models = {
        'KNeighborsRegressor': KNeighborsRegressor()
    }

    nn_models = {
        'MLPRegressor': MLPRegressor()
    }

    models = {
        **linear_models,
        **svr_models,
        **ensemble_models,
        **neighbors_models,
        **nn_models
    }
    return models