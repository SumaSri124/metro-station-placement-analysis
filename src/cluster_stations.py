import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("../data/processed_dataset.csv")

top_data = data.sort_values("score", ascending=False).head(1000)

k = 8
coords = top_data[["x", "y"]]

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(coords)

centers = kmeans.cluster_centers_

stations = pd.DataFrame(centers, columns=["x", "y"])
stations.to_csv("../data/predicted_stations.csv", index=False)

print("Predicted station locations saved")

plt.figure(figsize=(6,6))

plt.scatter(data["x"], data["y"], c=data["score"], s=10)
plt.scatter(stations["x"], stations["y"], color="red", s=100)

plt.title("Predicted Metro Stations")
plt.savefig("../outputs/station_map.png", dpi=300)
plt.show()
