import pandas as pd

df = pd.read_csv("/Users/zachpinto/Desktop/dev/silhouette/zip-code-map/data/cleaned_data.csv")

# Define the columns of interest
columns_of_interest = ['age_median', 'family_size', 'income_household_median', 'home_value', 'rent_median']

# Initialize an empty dictionary to store the results
summary_stats = {}

# Loop over each column of interest
for col in columns_of_interest:
    # Calculate the mean and median
    mean = df[col].mean()
    median = df[col].median()

    # Store the results in the dictionary
    summary_stats[col] = {"mean": mean, "median": median}

# Print the results
for col, stats in summary_stats.items():
    print(f"Column: {col}\nMean: {stats['mean']}\nMedian: {stats['median']}\n")
