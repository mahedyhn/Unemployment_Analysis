import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. Load the Data
# Using 'skipinitialspace' to handle potential spacing issues in CSV headers
df = pd.read_csv('unemployment_data.csv', skipinitialspace=True)

# 2. Data Cleaning
# Remove empty rows (those containing only commas)
df = df.dropna(how='all')

# Clean column names (strip leading/trailing spaces)
df.columns = df.columns.str.strip()

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Convert Area to category for better processing
df['Area'] = df['Area'].astype('category')

print("Data Cleaning Complete. Info:")
print(df.info())

# 3. Exploratory Data Analysis (EDA)
# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Average Unemployment Rate by Region
region_unemployment = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).reset_index()

# 4. Visualizations

# Chart 1: Unemployment Rate over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='Estimated Unemployment Rate (%)', hue='Area')
plt.title('Unemployment Rate Trends (Rural vs Urban)')
plt.xticks(rotation=45)
plt.show()

# Chart 2: Regional Impact (Average)
fig = px.bar(region_unemployment, x='Region', y='Estimated Unemployment Rate (%)', 
             title='Average Unemployment Rate by Region', color='Region')
fig.show()

# 5. Investigating COVID-19 Impact
# Defining Lockdown Period (April 2020 onwards in the dataset)
lockdown_df = df[df['Date'] >= '2020-04-01']
pre_lockdown_df = df[df['Date'] < '2020-04-01']

print("\n--- COVID-19 Impact Analysis ---")
print(f"Average Unemployment Rate PRE-Lockdown: {pre_lockdown_df['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"Average Unemployment Rate DURING-Lockdown: {lockdown_df['Estimated Unemployment Rate (%)'].mean():.2f}%")

# Chart 3: Heatmap of correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, 3:6].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Unemployment, Employment, and Participation')
plt.show()

# 6. Sunburst Plot for Area-wise and Region-wise Rate
fig = px.sunburst(df, path=['Area', 'Region'], values='Estimated Unemployment Rate (%)',
                  title='Unemployment Rate Distribution by Area and Region')
fig.show()