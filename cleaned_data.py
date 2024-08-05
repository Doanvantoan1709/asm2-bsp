import pandas as pd

# Load the CSV file
file_path = 'BSP.csv'
df = pd.read_csv(file_path)


# Checking for missing values in the columns 'Access Date', 'Page Views', and 'Time Spent'
missing_values = df[['Access Date', 'Page Views', 'Time Spent']].isnull().sum()
print("Missing values before cleaning:")
print(missing_values)


# Fill missing 'Access Date' with the mode
df['Access Date'].fillna(df['Access Date'].mode()[0], inplace=True)


# Fill missing 'Page Views' and 'Time Spent' with the median
df['Page Views'].fillna(df['Page Views'].median(), inplace=True)
df['Time Spent'].fillna(df['Time Spent'].median(), inplace=True)



# Check if there are any missing values left
missing_values_after = df[['Access Date', 'Page Views', 'Time Spent']].isnull().sum()
print("Missing values after cleaning:")
print(missing_values_after)



# Save the cleaned dataframe to a new CSV file
cleaned_file_path = 'Cleaned_BSP.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"Cleaned data saved to {cleaned_file_path}")
