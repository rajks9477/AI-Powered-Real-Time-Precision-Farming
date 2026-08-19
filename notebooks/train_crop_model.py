import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("/home/claude/ArogyaKrishi/data/Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15)
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("Accuracy:", accuracy_score(yte, pred))

joblib.dump(model, "/home/claude/ArogyaKrishi/models/RandomForest.pkl")
print("Saved model. Classes:", list(model.classes_))
