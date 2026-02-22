import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/processed_dataset.csv")

features = ["population", "traffic", "facilities", "centrality"]
X = data[features]
y = data["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

r2 = model.score(X_test, y_test)
print("R2 score:", r2)

xgb.plot_importance(model)
plt.title("Feature importance")
plt.savefig("../outputs/feature_importance.png", dpi=300)
plt.show()
