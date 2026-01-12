import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
data["Outcome"].value_counts().plot(kind="bar")
plt.title("Diabetes Outcome Count")
plt.show()
import matplotlib.pyplot as plt
diabetic = data[data["Outcome"] == 1]
non_diabetic = data[data["Outcome"] == 0]
plt.hist(non_diabetic["Glucose"], bins=20, alpha=0.7, label="Non-Diabetic")
plt.hist(diabetic["Glucose"], bins=20, alpha=0.7, label="Diabetic")
plt.xlabel("Glucose Level")
plt.ylabel("Number of Patients")
plt.title("Glucose Level vs Diabetes")
plt.legend()
plt.show()


