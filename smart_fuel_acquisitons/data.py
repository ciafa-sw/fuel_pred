from typing import Dict, Tuple, Union, Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

# get all days between 2 dates
# get the last day of the month of a given date
def get_last_day_of_month(date):
    return pd.Timestamp(date).replace(day=pd.Timestamp(date).days_in_month)

def get_days_between_dates(start_date, end_date):
    return pd.date_range(start_date, end_date, freq='D')

def dates_in_list(dates, date_list):
    return dates[[date in date_list for date in dates]]

def create_range_until_month_end(day: pd.Timestamp) -> pd.DatetimeIndex:
    '''
    This function will take in a day and return a range of business days from that day until the end of the month.
    '''
    month_ends = get_last_day_of_month(day)
    return pd.date_range(day, month_ends, freq='B')

def get_X_y_for_day(data: pd.DataFrame, day: pd.Timestamp,
                    window_size: int=30, max_horizon: int = 15,
                    target_column: int = 0) -> Tuple[np.array, np.array]:
    '''
    inputs:
        data: DataFrame with features (columns) and a datetime index
        day: the day to predict
        window_size: the number of lagged days to use as features
        max_horizon: the number of days to predict
        target_column: the index of the column to predict
    outputs:
        X: DataFrame with the lagged features
        ys: a dictionary with the horizon days as keys and the target values as values

    '''
    day_idx = data.index.get_loc(day)

    X = data.iloc[day_idx-window_size:day_idx]
    ys = {h: data.loc[day+pd.Timedelta(days=h)].iloc[:, target_column] for h in range(1, max_horizon + 1)}

    return X, ys


# Função para criar features e targets com múltiplos horizontes
def create_features_and_targets(data: pd.DataFrame, max_horizon: int=10,
                                window_size: int=30, target_column: int = 0) -> Tuple[np.array, Dict[int, np.array], Dict[int, pd.DatetimeIndex]]:
    '''
    inputs:
        data: np.array with the time series data, with several columns
        max_horizon: int with the number of horizons days to predict
        window_size: int with the number of lagged days to use, e.g. use last 30 days
        target_column: int with the index of the column to predict
    outputs:
        Xh: np.array with the lagged features
        yh: a dictionary with the horizon days as keys and the target values as values
        horizon_days: a dictionary with the horizon days as keys and the datetime index as values
    '''
    first_pred_day = data.index[window_size]
    last_pred_day = data.index[-max_horizon-1]
    
    pred_days = pd.date_range(first_pred_day, last_pred_day, freq='D')
    pred_days = dates_in_list(pred_days, data.index) # filter pred days not in data
    

    Xh = []
    yh: Dict[int, list] = {h: [] for h in range(0, max_horizon + 1)}
    horizon_days = {h: [] for h in yh.keys()}

    for day in pred_days:
        day_idx = data.index.get_loc(day)
        X = data.iloc[day_idx-window_size:day_idx]
        Xh.append(X.values.flatten())

        for h, y_lst in yh.items():
            y = data.iloc[day_idx + h, 0]

            y_lst.append(y)
            horizon_days[h].append(data.index[day_idx + h])

    Xh = np.array(Xh)
    yh = {h: np.array(y) for h, y in yh.items()}
    horizon_days = {h: pd.DatetimeIndex(hd) for h, hd in horizon_days.items()}

    return Xh, yh, horizon_days
    
    features_targets = {}
    for horizon_days in range(1, max_horizon + 1):
        X, y = [], []
        for i in range(len(data) - window_size - horizon_days + 1):
            X.append(data[i:i+window_size].flatten())
            y.append(data[i+window_size+horizon_days-1][target_column])  # O target é o preço do WTI Crude Oil
        features_targets[horizon_days] = (np.array(X), np.array(y))
    return features_targets



def create_features_for_month(data: pd.DataFrame, prediction_day: pd.Timestamp, window_size: int=30) -> pd.DataFrame:
    '''
    This function will take in data as a Pandas DataFrame with datetime index.
    It will also take a mask that has True on the days that we wish to predict.
    It will return the features with the window_size specified.
    '''
    assert isinstance(prediction_day, pd.Timestamp), f'prediction_day must be a string or a pd.Timestamp, got {type(prediction_day)}'
    

    expected_start = prediction_day - pd.offsets.BDay(window_size)
    expected_dt_range = pd.date_range(start=expected_start, end=prediction_day - pd.offsets.BDay(1), freq='B')

    # assert 30 days in expected_dt_range
    assert len(expected_dt_range) == window_size, f'Expected 30 days in the expected range, got {len(expected_dt_range)}'

    iloc_pred = data.index.get_loc(prediction_day)
    X = data.iloc[iloc_pred-window_size:iloc_pred]

    assert len(X) == window_size, f'Expected 30 days in the data fetched range, got {len(X)}'

    # send warning if start date of expected is different from one fethed from data
    
    if not X.index.equals(expected_dt_range):
        # compute the first date that is different, starting from end
        for i, (expected, fetched) in enumerate(zip(expected_dt_range[::-1], X.index[::-1])):
            if expected != fetched:
                break
        warnings.warn(f'Expected date range is different from fetched data. The first different dates are {expected} vs. {fetched}.')

    return X





def create_targets_for_month(data: pd.DataFrame, prediction_day: pd.Timestamp) -> pd.DataFrame:
    month_ends = get_last_day_of_month(prediction_day)
    y = data.loc[prediction_day:month_ends].iloc[:, 0]

    expect_future_range = create_range_until_month_end(prediction_day)
    
    if not y.index.equals(expect_future_range):
        # compute the first date that is different, starting from end
        for i, (expected, fetched) in enumerate(zip(expect_future_range[::-1], y.index[::-1])):
            if expected != fetched:
                break
        warnings.warn(f'Expected future date range is different from fetched data. The first different dates are {expected} vs. {fetched}.')
    return y

def create_features_and_targets_for_month(data: pd.DataFrame, prediction_day: pd.DatetimeIndex, window_size: int=30, target_column: int = 0) -> Tuple[np.array, np.array, pd.DatetimeIndex]:
    '''
    This function will take in data as a Pandas DataFrame with datetime index.
    It will also take a mask that has True on the days that we wish to predict.
    It will return the features with the window_size specified. The targets will be all the days from the prediction day until the end of the month.
    It will return the features and targets for the days we wish to predict.
    '''

    X = create_features_for_month(data, prediction_day, window_size)
    y = create_targets_for_month(data, prediction_day)

    return X.values, y.values, y.index


def predict_until_month_end(models: Dict[int, RegressorMixin], X, prediction_day) -> pd.DataFrame:
    future_days = create_range_until_month_end(prediction_day)
    ys = []
    for h in range(0, len(future_days)):
        model = models[h] # get model for that horizon
        # assert number of features is the same as the model
        assert len(X) == model.input_shape[1], f'Expected {model.input_shape[1]} features, got {len(X)}'
        y = model.predict(X) 
        ys.append(y)

    return pd.DataFrame(ys, index=future_days)
    

def predict_until_end(models: Dict[int, RegressorMixin], data: pd.DataFrame, prediction_day: Union[str, pd.Timestamp],
                      window_size: int, target_column: int):
    # prediction days
    future_days = create_range_until_month_end(prediction_day)
    if len(future_days) > max(models.keys()):
        raise ValueError(f'The greatest prediciton horizon is {max(models.keys())}, but there are {len(future_days)} business days to predict until the end of the month, since {prediction_day}.')

    X, y, dt = create_features_and_targets_for_month(data, prediction_day, window_size, target_column)

    X = X.flatten()

    ys = []
    for h in range(0, len(y)):
        model = models[h]
        ys.append(model.predict(X.reshape(1,-1)))

    return y, np.array(ys).flatten(), dt



def predict_month_avg(models: Dict[int, RegressorMixin], data: pd.DataFrame, prediction_day: Union[str, pd.Timestamp], window_size, target_column) -> float:
    prediction_day = pd.Timestamp(prediction_day)

    # check that prediction_day is a business day
    #month_business_days = pd.date_range(prediction_day, get_last_day_of_month(prediction_day), freq='B')
    #assert prediction_day in month_business_days, f'prediction_day must be a business day, got {prediction_day}'


    y, ys, dt = predict_until_end(models, data, prediction_day, window_size, target_column)

    month_begins = pd.Timestamp(prediction_day).replace(day=1)
    month_prices = data.loc[month_begins:prediction_day].iloc[:, target_column]

    predicted_prices = ys

    n_days = len(month_prices) + len(predicted_prices)

    return (month_prices.sum() + predicted_prices.sum()) / n_days


def create_features_and_targets_on_days_for_month(data: pd.DataFrame, prediction_days: pd.DatetimeIndex, window_size: int=30, target_column: int = 0) -> Tuple[np.array, np.array]:
    '''
    This function will take in data as a Pandas DataFrame with datetime index.
    It will also take a mask that has True on the days that we wish to predict.
    It will return the features with the window_size specified. The targets will be all the days from the prediction day until the end of the month.
    It will return the features and targets for the days we wish to predict.
    '''

    month_ends = pd.DatetimeIndex([get_last_day_of_month(d) for d in prediction_days])
    days_between = [get_days_between_dates(d, end) for d, end in zip(prediction_days, month_ends)]

    X = []
    ys = []

    for pred_day, days in zip(prediction_days, days_between):
        iloc_pred = data.index.get_loc(pred_day)

        X_ = data.iloc[iloc_pred-window_size:iloc_pred]
        y_ = data.loc[pred_day:days[-1]].iloc[:, 0]

        X.append(X_.values.flatten())
        ys.append(y_.values.flatten())
    return X, ys


