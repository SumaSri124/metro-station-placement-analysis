import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("city_dataset.csv")

center_x = 5
center_y = 5

data["distance_to_center"] = np.sqrt(
    (data["x"] - center_x)**2 +
    (data["y"] - center_y)**2
)

data["centrality"] = 1 / (data["distance_to_center"] + 1e-5)

# normalize features
features = ["population", "traffic", "facilities", "centrality"]

scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

#suitability score
data["score"] = (
    0.4 * data["population"] +
    0.25 * data["traffic"] +
    0.2 * data["facilities"] +
    0.15 * data["centrality"]
)

data.to_csv("processed_dataset.csv", index=False)

print("processed_dataset.csv saved")