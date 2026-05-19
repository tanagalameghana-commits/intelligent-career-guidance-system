import pandas as pd
import numpy as np
import os
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_PATH = r"C:\Users\tanag\Downloads\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main"

data_path = os.path.join(BASE_PATH, "dataset9000.data")

dataset = pd.read_csv(data_path, header=None)

X = np.array(dataset.iloc[:, 0:17])
Y = np.array(dataset.iloc[:, 17])

seed = 5
kfold = model_selection.KFold(n_splits=15, random_state=seed)

base_cls = DecisionTreeClassifier()

model = BaggingClassifier(
    estimator=base_cls,
    n_estimators=50,
    random_state=seed
)

results = model_selection.cross_val_score(model, X, Y, cv=kfold)

print("Accuracy :", results.mean() * 100)
model.fit(X, Y)

with open("career_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")