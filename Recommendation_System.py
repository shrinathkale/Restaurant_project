import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = "dataset.csv"
data = pd.read_csv(file_path)

data["Cuisines"].fillna("unknown")

categorical_columns = ["City", "Cuisines", "Has Table booking", "Has Online delivery"]

label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

def user_pref(user_cuisine,user_price_range,user_min_rating,top_n=5):
    if user_cuisine in label_encoders["Cuisines"].classes_:
        user_cuisine_encoded = label_encoders["Cuisines"].transform([user_cuisine])[0]
    else:
        print("Cuisine not found in the dataset")
        return pd.DataFrame()
    
    # Filter restaurants based on user preferences
    filtered_data = data[(data['Cuisines'] == user_cuisine_encoded) & (data['Price range'] <= user_price_range) & (data['Aggregate rating'] >= user_min_rating)]

    recommendation = filtered_data.sort_values(by=["Aggregate rating","Votes"], ascending=[False,False]).head(top_n)

    recommendation["Cuisines"] = label_encoders["Cuisines"].inverse_transform(recommendation["Cuisines"])
    recommendation["City"] = label_encoders["City"].inverse_transform(recommendation["City"])

    return recommendation[["City","Restaurant Name","Cuisines","Price range","Aggregate rating","Votes"]]

user_cuisine = input("Enter cuisine name (1st letter must capital): ")
user_price_range = int(input("Enter price range (1-4): "))
user_min_rating = float(input("Enter minimum rating (1-5): "))

recomm = user_pref(user_cuisine,user_price_range,user_min_rating)
print(recomm)





# OUTPUT

# Enter cuisine name (1st letter must capital):  Japanese
# Enter price range (1-4):  3
# Enter minimum rating (1-5):  4

#                 City   Restaurant Name  Cuisines  Price range  \
# 429   Rest of Hawaii     Marukame Udon  Japanese            1   
# 1        Makati City  Izakaya Kikufuji  Japanese            3   
# 222           Dalton      Soho Hibachi  Japanese            1   
# 9289         Jakarta    3 Wise Monkeys  Japanese            3   

#       Aggregate rating  Votes  
# 429                4.9    602  
# 1                  4.5    591  
# 222                4.3    116  
# 9289               4.2    395  