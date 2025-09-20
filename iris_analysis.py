# Assignment: Analyzing Data with Pandas and Visualizing Results with Matplotlib

# ==========================
# Task 1: Load & Explore Data
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target  # Add species column (numeric codes)

# Show first few rows
print("First 5 rows of the dataset:")
print(df.head(), "\n")

# Info about dataset
print("Dataset Info:")
print(df.info(), "\n")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum(), "\n")


# ==========================
# Task 2: Basic Data Analysis
# ==========================

# Basic statistics
print("Statistical Summary:")
print(df.describe(), "\n")

# Group by species and get mean sepal length
print("Mean Sepal Length by Species:")
print(df.groupby("species")["sepal length (cm)"].mean(), "\n")

# Example finding:
# Species 2 generally has larger petals compared to species 0 and 1.


# ==========================
# Task 3: Data Visualization
# ==========================

# 1. Line Chart (Sepal Length trend across dataset indices)
plt.plot(df.index, df["sepal length (cm)"])
plt.title("Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.show()

# 2. Bar Chart (Average Petal Length per Species)
df.groupby("species")["petal length (cm)"].mean().plot(kind="bar")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 3. Histogram (Distribution of Sepal Width)
df["sepal width (cm)"].hist()
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (Sepal Length vs Petal Length, colored by species)
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], c=df["species"], cmap="viridis")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.colorbar(label="Species")
plt.show()


# ==========================
# Observations
# ==========================
# - Species 0 tends to have shorter petals compared to species 1 and 2.
# - Sepal length shows an increasing trend across the dataset indices.
# - The histogram shows sepal width is roughly normally distributed.
# - Scatter plot reveals a clear separation between species based on sepal and petal length.
