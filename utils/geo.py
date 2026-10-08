import numpy as np

EARTH_RADIUS_M = 6371008.8

def haversine_distance(pred, true):
    """Great-circle distance in metres between arrays of (LATITUDE, LONGITUDE) in degrees, shape [N, 2]."""
    lat1, lon1 = np.radians(pred[:, 0]), np.radians(pred[:, 1])
    lat2, lon2 = np.radians(true[:, 0]), np.radians(true[:, 1])
    a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))

def to_local_m(latlon):
    """
    (LATITUDE, LONGITUDE) in degrees -> planar (East, North) coordinates in metres
    (equirectangular projection), so Euclidean distances are distances in metres.
    """
    latlon = np.asarray(latlon, dtype=float)
    lat, lon = np.radians(latlon[:, 0]), np.radians(latlon[:, 1])
    return np.c_[EARTH_RADIUS_M * lon * np.cos(lat), EARTH_RADIUS_M * lat]

class GeoRegressor:
    """
    Wraps a coordinate -> RSRP regressor so that it is trained and queried on
    (LATITUDE, LONGITUDE) in degrees but works internally on planar metres.
    """
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(to_local_m(X), y)
        self.y_train_ = np.asarray(y)
        return self

    def predict(self, X):
        return self.model.predict(to_local_m(X))

    def kneighbors(self, X, n_neighbors=None):
        return self.model.kneighbors(to_local_m(X), n_neighbors)
