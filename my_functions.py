import pandas as pd
import numpy as np
from sklearn import cluster

def add_datetime_features(df):
    """
    The function takes as input a table with data on cab rides (DataFrame) and 
    returns the same table with 3 columns added to it:
    * pickup_date - the start of the trip (date without time);
    * pickup_hour - hour when the counter was activated;
    * pickup_day_of_week - the day of the week when the counter was activated
    
    Args:
        df (pandas.DataFrame):  table with cab rides data;
        
    Returns:
        pandas.DataFrame: table with cab rides data with three new columns
    """
    df['pickup_date'] = df['pickup_datetime'].dt.date
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df

def add_holiday_features(holiday_data, taxi_data):
    """
    The function takes as input two tables with data on cab rides (DataFrame) and holidays (DataFrame) and
    returns the same table as the cab rides one with 2 columns added to it:
    * pickup_holiday added - a binary feature that indicates whether 
                            a trip is started on a holiday or not (1 - yes, 0 - no). 

    Args:
        holiday_data (pandas.DataFrame): table with cab rides data;
        taxi_data (pandas.DataFrame): table with holiday data;
        
    Returns:
        pandas.DataFrame: table with cab rides data with one new column
    """
    holiday_data['date'] = pd.to_datetime(holiday_data['date'])
    holiday_data['holiday'] = 1
    taxi_data = taxi_data.merge(holiday_data, left_on='pickup_date', right_on='date', how='left')
    taxi_data['pickup_holiday'] = taxi_data['holiday'].fillna(0)
    taxi_data.drop(columns=['day', 'date', 'holiday'], inplace=True)
    return taxi_data

def add_osrm_features(taxi_data, osrm_data):
    """
    The function takes as input two tables with data on cab rides (DataFrame) and on OSRM
    returns the same table with 3 columns added to it:
    * total_distance;
    * total_travel_time;
    * number_of_steps.

    Args:
        taxi_data (pandas.DataFrame): table with cab rides data;
        osrm_data (_type_): table with data from OSRM API;

    Returns:
        pandas.DataFrame: table with cab rides data with 3 added columns
    """  
    taxi_data = taxi_data.merge(osrm_data[['total_distance', 'total_travel_time', 'number_of_steps', 'id']], left_on='id', right_on='id', how='left')
    return taxi_data


def get_haversine_distance(lat1, lng1, lat2, lng2):
    """
    The function takes as input four pd.Series with pick-up and drop-off coordinates and
    returns a pd.Series with Haversine distance:
    Args:
        lat1 (pandas.Series): pickup latitude		
        lng1 (pandas.Series): pickup longitude
        lat2 (pandas.Series): dropoff latitude
        lng2 (pandas.Series): dropoff longitude

    Returns:
        pandas.Series: haversine distance
    """
    # transform angles to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # radius of the Earth in km
    EARTH_RADIUS = 6371 
    # calculate shortest distance based on Haversine formula
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def get_angle_direction(lat1, lng1, lat2, lng2):
    """
    The function takes as input four pd.Series with pick-up and drop-off coordinates and
    returns a pd.Series with angle of the direction:

    Args:
        lat1 (pandas.Series): pickup latitude;		
        lng1 (pandas.Series): pickup longitude;
        lat2 (pandas.Series): dropoff latitude;
        lng2 (pandas.Series): dropoff longitude.

    Returns:
        pandas.Series: angle of the direction
    """
    # transform angles to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    #  calculate the angle of the direction, alpha, according to the Bearing formula
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha

def add_geographical_features(taxi_data,coordinates):
    """
    The function takes as input a table with data on cab rides (DataFrame) and 
    the list of the coordinate features and returns the table with two added columns:
    * haversine_distance - Haversine distance btw pick-up and drop-off locations
    * direction - angle btw pick-up and drop-off locations
    Args:
        taxi_data (pandas.DataFrame): 
        coordinates (list): a list with the coordinate features names, namely
                            pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
    Returns:
        pandas.DataFrame: table with cab rides data with 2 new columns
    """
    taxi_data['haversine_distance'] =  taxi_data[coordinates].apply(lambda x: get_haversine_distance(*x), axis=1)
    taxi_data['direction'] = taxi_data[coordinates].apply(lambda x: get_angle_direction(*x), axis=1)
    return taxi_data


def add_weather_features(taxi_data, weather_data):
    """
    The function takes as input two tables with data on cab rides (DataFrame) and on weather (DataFrame) and
    returns the same table as the cab rides one with 5 columns added to it:
    * temperature;
    * visibility;
    * wind speed;
    * precip;
    * events.

    Args:
        taxi_data (pandas.DataFrame):  table with holiday data;
        weather_data (pandas.DataFrame): table with weather data

    Returns:
        pandas.DataFrame:  table with cab rides data with five new columns
    """
    weather_data['time'] = pd.to_datetime(weather_data['time'])
    weather_data['date'] = weather_data['time'].dt.date
    weather_data['hour'] = weather_data['time'].dt.hour
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    weather_data = weather_data [['temperature', 'visibility', 'wind speed', 'precip', 'events', 'date', 'hour']]
    taxi_data = taxi_data.merge(weather_data, left_on=['pickup_date', 'pickup_hour'],  right_on=['date', 'hour'], how='left')
    return taxi_data

coordinates_pickup = ['pickup_latitude', 'pickup_longitude']
coordinates_dropoff = ['dropoff_latitude', 'dropoff_longitude']

def add_cluster_features(taxi_data, coordinates_pickup, coordinates_dropoff, 
                         n_clusters=10, random_state = 42): 
    """Реализуйте функцию add_cluster_features(), которая принимает на вход таблицу с данными о поездках 
    и обученный алгоритм кластеризации. Функция должна возвращать обновленную таблицу с добавленными 
    в нее столбцом geo_cluster - географический кластер, к которому относится поездка.

    Args:
        taxi_data (pandas.DataFrame):  table with holiday data;
        coordinates_pickup (list): a list with the coordinate features names, namely
                            pickup_latitude, pickup_longitude
        coordinates_dropoff (list): a list with the coordinate features names, namely
                            dropoff_latitude, dropoff_longitude
        n_clusters (int, optional): number of clusters for KMeans. Defaults to 10.
        random_state (int, optional): random state for KMeans. Defaults to 42.

    Returns:
        pandas.DataFrame:  table with cab rides data with one new column
    """
    coords = np.hstack((taxi_data[coordinates_pickup],
                    taxi_data[coordinates_dropoff]))
    # clustering algorithms
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_alg= kmeans.fit(coords)
    taxi_data['geo_cluster'] = cluster_alg.predict(coords)
    return taxi_data

