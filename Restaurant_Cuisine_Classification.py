import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

file_path = "dataset.csv"
data = pd.read_csv(file_path)

data = data.dropna(subset=["Cuisines"])
data["pri cuisine"] = data["Cuisines"].apply(lambda x: x.split(",")[0])

features = ["City","Price range","Average Cost for two","Aggregate rating","Votes"]

x = data[features]
y = data["pri cuisine"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["pri cuisine"])

numeric_features = ["Price range","Average Cost for two","Aggregate rating","Votes"]
categorical_features = ["City"]

numeric_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("standard", StandardScaler())])
categorical_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}

loo = LeaveOneOut()
grid_search = GridSearchCV(model, param_grid, cv=loo, scoring="accuracy")
grid_search.fit(x_train, y_train)

# Evaluate
y_pred = grid_search.best_estimator_.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)