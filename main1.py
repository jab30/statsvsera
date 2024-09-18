import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st
import numpy as np

# Fetch player statistics data from Fangraphs API for the 2024 season
@st.cache_data
def load_data():
    url = "https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&season=2024&season1=2024&ind=0&qual=0&type=8&month=0&pageitems=500000"
    data = requests.get(url).json()
    df = pd.DataFrame(data=data['data'])
    return df

# Load the data
df = load_data()

# Ensure the DataFrame contains the required columns
required_columns = ['FBv', 'FIP', 'IP', 'LA', 'Barrels', 'Barrel%', 'maxEV', 'HardHit',
                    'HardHit%', 'K%', 'BB%', 'K-BB%', 'SIERA', 'O-Swing%', 'Z-Swing%',
                    'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%',
                    'SwStr%', 'CStr%', 'C+SwStr%', 'AVG', 'WHIP', 'BABIP', 'LD%', 'GB%',
                    'FB%', 'ERA', 'WAR']

# Raise an error if any required column is missing
for col in required_columns:
    if col not in df.columns:
        st.error(f"The required column '{col}' is not present in the DataFrame.")
        st.stop()

# Convert relevant columns to numeric
df[required_columns] = df[required_columns].apply(pd.to_numeric, errors='coerce')

# Filter pitchers with a minimum of 50 IP
df_filtered = df[df['IP'] >= 50].dropna(subset=required_columns)

# Function to plot the selected stat
def plot_stat(stat):
    plt.figure(figsize=(10, 6))
    plt.scatter(df_filtered[stat], df_filtered['ERA'], alpha=0.7, color='blue')

    # Fit a linear regression model to get R²
    X = df_filtered[[stat]]
    y = df_filtered['ERA']
    model = LinearRegression()
    model.fit(X, y)

    # Predict ERA based on the selected stat
    y_pred = model.predict(X)

    # Calculate R²
    r_squared = model.score(X, y)

    # Plot the regression line
    plt.plot(df_filtered[stat], y_pred, color='red', linewidth=2)

    # Annotate R² on the plot
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

    # Add plot title and labels
    plt.title(f'{stat} vs ERA (Pitchers with minimum 50 IP)', fontsize=16)
    plt.xlabel(stat, fontsize=14)
    plt.ylabel('Earned Run Average (ERA)', fontsize=14)
    plt.grid(True)

    st.pyplot(plt)

# Streamlit UI for dropdown
st.title("Pitcher Stats vs ERA")
stat = st.selectbox('Select Stat', required_columns[3:], index=0)

# Plot the selected stat
plot_stat(stat)
