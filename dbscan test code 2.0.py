import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
file_path = 'csv_cleaned_marketing_campaign.csv'  # Update this to the actual file path
df = pd.read_csv(file_path)

# Summing all columns starting with "Mnt" for total spending
mnt_cols = [col for col in df.columns if col.startswith('Mnt')]
df['Total_Spending'] = df[mnt_cols].sum(axis=1)

# Summing 'Kidhome' and 'Teenhome' for total number of children
df['Total_Children'] = df['Kidhome'] + df['Teenhome']

# Selecting the new features and handling missing values
data = df[['Income', 'Total_Spending', 'Total_Children']].dropna()

# Normalizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Running DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
clusters = dbscan.fit_predict(data_scaled)

# Using PCA for visualization (reducing to 2D space for plotting)
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data_scaled)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=clusters, cmap='viridis', label='Customer Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('DBSCAN Clustering with Sum of Spending and Children')
plt.colorbar(label='Cluster Label')
plt.show()
