import pandas as pd
import numpy as np
import pickle
import os

BASE_PATH = r"C:\Users\tanag\Downloads\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main\INTELLIGENT-CAREER-GUIDANCE-SYSTEM-main"

data_path = os.path.join(BASE_PATH, "dataset9000.data")

career = pd.read_csv(data_path, header=None)

X = np.array(career.iloc[:, 0:17])
y = np.array(career.iloc[:, 17])

career.columns = [
    "Database Fundamentals","Computer Architecture","Distributed Computing Systems",
    "Cyber Security","Networking","Development","Programming Skills","Project Management",
    "Computer Forensics Fundamentals","Technical Communication","AI ML","Software Engineering",
    "Business Analysis","Communication skills","Data Science","Troubleshooting skills",
    "Graphics Designing","Roles"
]

career.dropna(how='all', inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=524
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Accuracy =", metrics.accuracy_score(y_test, y_pred) * 100)

model_path = os.path.join(BASE_PATH, "careerlast.pkl")
pickle.dump(knn, open(model_path, 'wb'))

print("Model saved successfully")