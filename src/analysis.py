from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analysis:
    
    @staticmethod
    def PCA(dataframe, n_components):
        # standardize the data and scale
        scaler = StandardScaler() 
        df_scaled = scaler.fit_transform(dataframe)

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df_scaled)

        # Create a DataFrame that will have
        columns = Analysis.columns(n_components)
        pca_df = pd.DataFrame(data=principal_components, columns=columns)

        return (pca.explained_variance_ratio_, pca_df)
    


    @staticmethod   
    def plot_PCA(pca_df, dataset_name, coloring_feature):

        plt.figure(figsize=(10, 6))

        if coloring_feature is not None:
            pca_df[coloring_feature[1]] = coloring_feature[0]  # Add the coloring feature
            # Normalize the E033 values for color mapping
            norm = plt.Normalize(coloring_feature[0].min(), coloring_feature[0].max())
            cmap = plt.get_cmap('coolwarm')  # Get the colormap from blue to red

            # Map E033 values to colors
            colors = cmap(norm(coloring_feature[0]))
            scatter = plt.scatter(x=pca_df['PC1'], y=pca_df['PC2'],alpha=0.7, c=colors)
            plt.title(f'PCA Plot of {dataset_name} Colored by {coloring_feature[1]} ({coloring_feature[2]})')

        else:
            scatter = plt.scatter(x=pca_df['PC1'], y=pca_df['PC2'],alpha=0.7)
        plt.title(f'PCA Plot of {dataset_name}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid()
        plt.show()


    @staticmethod
    def columns(n_components):
        return [f'PC{i}' for i in range(1, n_components+1)]

    