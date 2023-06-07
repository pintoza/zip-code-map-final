import pandas as pd

df = pd.read_csv("/Users/zachpinto/Desktop/dev/silhouette/zip-code-map/data/cleaned_data.csv")

# Remove the last five rows
df = df.iloc[:-5]

# Convert zip from float to int
df['zip'] = df['zip'].astype(int)

# Convert zip from int to string
df['zip'] = df['zip'].astype(str)

# Apply zfill to ensure all zip codes have 5 digits
df['zip'] = df['zip'].apply(lambda x: x.zfill(5))

# List of columns to convert from decimal to integers
columns_to_convert = ["population"]

# Convert columns from float to int
for column in columns_to_convert:
    df[column] = df[column].astype(int)

# Remove the columns you don't want
df = df.drop(['pop_per_sq_mi', 'rucc', "# of ACCOUNTS", "UNITS ROLL. 12"], axis=1)

# List of columns to convert from decimal to actual numbers based on their relation to population
columns_to_convert = ['race_multiple', 'race_other', 'race_pacific', 'race_native', 'race_asian', 'race_black',
                      'race_white', 'education_college_or_above', 'married', 'female', 'male']

# Convert decimals to actual numbers
for column in columns_to_convert:
    df[column] = df[column]*df['population']

# Create a new column 'labor force'
df['labor_force'] = df['labor_force_participation']*df['population']

# Create a new column 'unemployed'
df['unemployed'] = df['unemployment_rate']*df['labor_force']

# Create a new column 'households'
df['households'] = df['population']/df['family_size']

# Create a new column 'home_own'
df['home_own'] = df['households']*df['home_ownership']

# Create a new column 'households_six_fig'
df['households_six_fig'] = df['households']*df['income_household_six_figure']

# Drop 'unemployment_rate' and 'labor_force_participation' columns
df = df.drop(['unemployment_rate', 'labor_force_participation',
              'home_ownership', 'income_household_six_figure'], axis=1)

# Replace all NaN values with 0
df.fillna(0, inplace=True)

# Round everything to the nearest integer except for "age_median" and "family_size" and convert to int
cols_to_exclude = ["age_median", "family_size", "zip", "lat", "lng", "city", "state_id", "state_name", "primary_county"]
for col in df.columns:
    if col not in cols_to_exclude:
        df[col] = df[col].round().astype(int)

# Save the final dataframe to a csv file
df.to_csv('/Users/zachpinto/Desktop/dev/silhouette/zip-code-map/data/data_final.csv', index=False)
