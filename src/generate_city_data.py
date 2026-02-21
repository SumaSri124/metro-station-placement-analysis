import numpy as np
import pandas as pd

# creating a simple "city-like" dataset
# each point represents a location in a city

np.random.seed(42)

num_points = 2000

# random coordinates in a 10x10 grid
x = np.random.uniform(0, 10, num_points)
y = np.random.uniform(0, 10, num_points)

# helper function to create dense areas (like city centers)
def make_cluster(x, y, cx, cy, spread):
    return np.exp(-((x - cx)**2 + (y - cy)**2) / spread)

# simulate population density
population = (
    3000 * make_cluster(x, y, 5, 5, 2) +
    2000 * make_cluster(x, y, 7, 3, 1.5) +
    1000 * np.random.rand(num_points)
)

# simulate traffic  
traffic = (
    100 * make_cluster(x, y, 5, 5, 3) +
    50 * np.random.rand(num_points)
)

# simulate facilities
facilities = (
    10 * make_cluster(x, y, 6, 6, 2) +
    5 * np.random.rand(num_points)
)

data = pd.DataFrame({
    "x": x,
    "y": y,
    "population": population,
    "traffic": traffic,
    "facilities": facilities
})

data.to_csv("city_dataset.csv", index=False)

print("Dataset saved as city_dataset.csv")