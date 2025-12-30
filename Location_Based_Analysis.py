import pandas as pd
import matplotlib.pyplot as plt

file_path = "dataset.csv"
data = pd.read_csv(file_path)

#Visualization of restaurant distribution according to longitude and latitude
def restaurant_distribution(data):
    plt.figure(figsize=(12,8))
    plt.scatter(data["Longitude"], data["Latitude"], c="red", alpha=0.5, s=10, label="Restaurants")
    plt.title("Visualization of Restaurant Distribution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

#Analysis of cities according to restaurant concentration
def restaurant_concentration(data):
    concentration = data.groupby("City").size().reset_index(name="Restaurant Count")
    concentration = concentration.sort_values(by="Restaurant Count", ascending=False)
    return concentration

#Calculation of statistics for cities
def city_statistics(data):
    statistics = data.groupby("City").agg({"Aggregate rating":"mean", "Price range":"mean", "Votes":"sum","Cuisines":lambda x: set(x)}).rename(columns={"Aggregate rating":"Average Rating", "Price range":"Average Price Range", "Votes":"Total Votes"}).reset_index()
    statistics = statistics.merge(data[["City","Country Code"]].drop_duplicates(), on="City", how="left")
    statistics = statistics.sort_values(by="Average Rating", ascending=False)
    return statistics

#Insight(conclusion) function
def insights(data):
    
    while True:
        print("1.Restaurant Distribution")
        print("2.Restaurant Concentration")
        print("3.City statistics")
        print("4.Exit")
        ch = int(input("Choose your option number: "))

        match ch:
            case 1:
                restaurant_distribution(data)         #function calling   
            case 2:
                conce = restaurant_concentration(data)      #function calling
                print("Top 5 cities with restaurants concentration")
                print(conce.head())
            case 3:
                stats = city_statistics(data)     #function calling
                print("Top 5 cities statistics")
                print(stats.head())
            case 4:
                break
            case default:
                print("Invalid choice")
                
insights(data)