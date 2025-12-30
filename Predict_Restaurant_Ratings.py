import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

file_path = "dataset.csv"
data = pd.read_csv(file_path)

features = ["Country Code","City","Cuisines","Average Cost for two","Has Table booking","Has Online delivery","Price range","Votes"]
target = "Aggregate rating"

x = data[features]
y = data[target]

#data preprocessing
numerical_features = ["Country Code","Average Cost for two","Price range","Votes"]
categorical_features = ["City","Cuisines","Has Table booking","Has Online delivery"]

numerical_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])

categorical_transformer = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, numerical_features), ("cat", categorical_transformer, categorical_features)])

model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Mean Squared Error: ",mse)
print("R-Squared Score: ",r2)

#feature importance
regressor = model.named_steps["regressor"]
feature_names = preprocessor.transformers_[0][1].get_feature_names_out(numerical_features).tolist() + preprocessor.transformers_[1][1].named_steps["onehot"].get_feature_names_out(categorical_features).tolist()

feature_importances = regressor.feature_importances_

feature_importance_df = pd.DataFrame({"Feature":feature_names, "Importance":feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

print("Feature Importance Analysis\n",feature_importance_df)

# Function for predicting on runtime user input
def predict_runtime_input(model, preprocessor):
    print("\nEnter values for prediction:\n")
    city = input("Enter city: ")
    cc = int(input("Country Code (numeric): "))
    cus = input("Cuisines (string): ")
    av = float(input("Average Cost for two (numeric): "))
    ht = input("Has Table booking (Yes/No): ")
    ho = input("Has Online delivery (Yes/No): ")
    pr = int(input("Price range (1-4): "))
    v = int(input("Votes (numeric): "))
    user_input = {"Country Code" : cc, "City" : city, "Cuisines" : cus, "Average Cost for two" : av, "Has Table booking" : ht, "Has Online delivery" : ho, "Price range" : pr,"Votes" : v}

    # Convert to DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Predict
    prediction = model.predict(user_input_df)
    print(f"\nPredicted Aggregate Rating: {prediction[0]:.2f}")

# Call the function to get predictions for runtime user input
predict_runtime_input(model, preprocessor)