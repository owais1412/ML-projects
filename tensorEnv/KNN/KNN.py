import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(data["buying"])
maint = le.fit_transform(data["maint"])
door = le.fit_transform(data["door"])
persons = le.fit_transform(data["persons"])
lug_boot = le.fit_transform(data["lug_boot"])
safety = le.fit_transform(data["safety"])
cls = le.fit_transform(data["class"])

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print(f"Predicted: {names[predicted[x]]} Data: {x_test[x]}, Actual: {names[y_test[x]]}")