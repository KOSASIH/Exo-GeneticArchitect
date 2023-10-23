import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Perform any necessary preprocessing steps on the genetic data
    # For example, you can normalize the data or handle missing values
    
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data

def perform_pca(data, n_components=2):
    # Perform Principal Component Analysis (PCA) on the data
    
    # Create a PCA object with the desired number of components
    pca = PCA(n_components=n_components)
    
    # Apply PCA on the data
    pca_data = pca.fit_transform(data)
    
    return pca_data

def perform_tsne(data, n_components=2, perplexity=30):
    # Perform t-Distributed Stochastic Neighbor Embedding (t-SNE) on the data
    
    # Create a t-SNE object with the desired number of components and perplexity
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    
    # Apply t-SNE on the data
    tsne_data = tsne.fit_transform(data)
    
    return tsne_data

def visualize_data(data, labels=None):
    # Visualize the reduced data using scatter plots or heatmaps
    
    # Create a scatter plot of the reduced data
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Dimensionality Reduction')
    plt.colorbar()
    plt.show()

# Example usage
genetic_data = np.array([[1.2, 3.4, 2.1, 4.5], [2.3, 4.5, 1.9, 3.2], [3.1, 2.5, 4.3, 1.8]])
preprocessed_data = preprocess_data(genetic_data)

# Perform PCA
pca_data = perform_pca(preprocessed_data)
visualize_data(pca_data)

# Perform t-SNE
tsne_data = perform_tsne(preprocessed_data)
visualize_data(tsne_data)
