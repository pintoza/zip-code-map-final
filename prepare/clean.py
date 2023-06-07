import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('/Users/zachpinto/Desktop/dev/silhouette/zip-code-map/data/map.csv')

# Make a copy of your original dataframe
df_original = df.copy()

# FILL IN MISSING MEDIAN AGE VALUES

# Define the features to be used
features = ['population', 'pop_per_sq_mi', 'male', 'female', 'married', 'family_size']

# Reset dataframe to original
df = df_original.copy()

# First, fill missing values in the features used for prediction with their median
for feature in features:
    df[feature].fillna(df[feature].median(), inplace=True)

# Now, proceed with the model-based imputation for 'Median Age'

# Split the data into two sets: data with known 'Median Age' and data with unknown 'Median Age'
known_age = df[df['age_median'].notnull()]
unknown_age = df[df['age_median'].isnull()]

# Prepare the data for training the model
X = known_age[features]
y = known_age['age_median']

# Split the data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predict the missing 'Median Age' values
predicted_ages = rf.predict(unknown_age[features])

# Fill in the missing values in the original dataframe
df_original.loc[df['age_median'].isnull(), 'age_median'] = predicted_ages

# FILL IN MISSING FAMILY SIZE VALUES

# Define the features to be used
features = ['population', 'pop_per_sq_mi', 'male', 'female', 'married', 'age_median']

# Reset dataframe to original
df = df_original.copy()

# First, fill missing values in the features used for prediction with their median
for feature in features:
    df[feature].fillna(df[feature].median(), inplace=True)

# Now, proceed with the model-based imputation for 'Family Size'

# Split the data into two sets: data with known 'Family Size' and data with unknown 'Family Size'
known_family_size = df[df['family_size'].notnull()]
unknown_family_size = df[df['family_size'].isnull()]

# Prepare the data for training the model
X = known_family_size[features]
y = known_family_size['family_size']

# Split the data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predict the missing 'Family Size' values
predicted_family_sizes = rf.predict(unknown_family_size[features])

# Fill in the missing values in the original dataframe
df_original.loc[df['family_size'].isnull(), 'family_size'] = predicted_family_sizes

# FILL IN MISSING HOUSEHOLD MEDIAN INCOME

# Define the features to be used
features = ['population', 'pop_per_sq_mi', 'male', 'female', 'married', 'age_median',
            'education_college_or_above', 'home_value', 'rent_median']

# Reset dataframe to original
df = df_original.copy()

# First, fill missing values in the features used for prediction with their median
for feature in features:
    df[feature].fillna(df[feature].median(), inplace=True)

# Now, proceed with the model-based imputation for 'Income Household Median'

# Split the data into two sets: data with known 'Income Household Median' and  with unknown 'Income Household Median'
known_income = df[df['income_household_median'].notnull()]
unknown_income = df[df['income_household_median'].isnull()]

# Prepare the data for training the model
X = known_income[features]
y = known_income['income_household_median']

# Split the data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predict the missing 'Income Household Median' values
predicted_incomes = rf.predict(unknown_income[features])

# Fill in the missing values in the original dataframe
df_original.loc[df['income_household_median'].isnull(), 'income_household_median'] = predicted_incomes

# FILL IN MISSING HOME VALUE

# Define the features to be used
features = ['population', 'pop_per_sq_mi', 'male', 'female', 'married', 'age_median',
            'education_college_or_above', 'income_household_median', 'rent_median']

# Reset dataframe to original
df = df_original.copy()

# First, fill missing values in the features used for prediction with their median
for feature in features:
    df[feature].fillna(df[feature].median(), inplace=True)

# Now, proceed with the model-based imputation for 'Home Value'

# Split the data into two sets: data with known 'Home Value' and data with unknown 'Home Value'
known_home_value = df[df['home_value'].notnull()]
unknown_home_value = df[df['home_value'].isnull()]

# Prepare the data for training the model
X = known_home_value[features]
y = known_home_value['home_value']

# Split the data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predict the missing 'Home Value' values
predicted_home_values = rf.predict(unknown_home_value[features])

# Fill in the missing values in the original dataframe
df_original.loc[df['home_value'].isnull(), 'home_value'] = predicted_home_values

# FILL IN MISSING RENT MEDIAN VALUES

# Define the features to be used
features = ['population', 'pop_per_sq_mi', 'male', 'female', 'married', 'age_median',
            'education_college_or_above', 'income_household_median', 'home_value']

# Reset dataframe to original
df = df_original.copy()

# First, fill missing values in the features used for prediction with their median
for feature in features:
    df[feature].fillna(df[feature].median(), inplace=True)

# Now, proceed with the model-based imputation for 'Rent Median'

# Split the data into two sets: data with known 'Rent Median' and data with unknown 'Rent Median'
known_rent = df[df['rent_median'].notnull()]
unknown_rent = df[df['rent_median'].isnull()]

# Prepare the data for training the model
X = known_rent[features]
y = known_rent['rent_median']

# Split the data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# Predict the missing 'Rent Median' values
predicted_rents = rf.predict(unknown_rent[features])

# Fill in the missing values in the original dataframe
df_original.loc[df['rent_median'].isnull(), 'rent_median'] = predicted_rents

# save data to csv within root/data
df_original.to_csv("/Users/zachpinto/Desktop/dev/silhouette/zip-code-map/data/cleaned_data.csv", index=False)
