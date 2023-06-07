import pandas as pd
import openpyxl
import os

# Define the path to your Excel file
excel_path = '/Users/zachpinto/Desktop/dev/silhouette/zip-code-map/data/map.xlsx'

# Load your data
df = pd.read_excel(excel_path)

# Save the dataframe as a CSV in the 'data' directory
df.to_csv('map.csv', index=False)
