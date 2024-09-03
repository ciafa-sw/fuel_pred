from typing import Callable, Tuple
import pandas as pd



def make_prediction_day_mask(data: pd.DataFrame, prediction_day: str = '2nd last wed') -> pd.Series:
    '''
    prediction_day is a string that specifies the day of the month to predict the fuel cost
    '''
    raise NotImplementedError



def compute_cost(model: Callable, data: pd.DataFrame, prediction_day: str = '2nd last wed') -> Tuple[float, float]:
    '''
    inputs:
        model: is a function that receives a pandas dataframe and predicts the monthly cost of fuel at the specified prediction day
        data: is a pandas dataframe with a DateTime index and the following columns:
            - 'WTI Price': a float with the price of fuel
            ...
    outputs:
        - a pd.Series with the real cost of fuel for each month
        - a pd.Series with the predicted cost of fuel for each month
    '''
    raise NotImplementedError