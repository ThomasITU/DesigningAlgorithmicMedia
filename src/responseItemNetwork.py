import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

class ResponseItemNetwork:
    def __init__(self, df):
        self.df = self.binarize_df(df)
        self.nodes = list(self.df.columns)
        self.edges = {}
        self.build_graph()

    def build_graph(self):
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node_a, node_b = self.nodes[i], self.nodes[j]
                correlation = self.calculate_correlation(node_a, node_b)
                if correlation > 0:  # Only add positive associations from the paper
                    self.add_edge(node_a, node_b, correlation)

    def binarize_df(self, df):
        # Create an empty DataFrame to store results
        binarized = pd.DataFrame(index=df.index)
        
        # Process each column individually
        for col in df.columns:
            # Get min and max values for this column
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Create new column name
            new_col = f"{col}"
            
            # Create binarized values
            binarized[new_col] = pd.Series(
                np.where(df[col] == min_val, 0,
                        np.where(df[col] == max_val, 1, -1)),
                index=df.index
            )
        
        return binarized
        

    def add_edge(self, source, target, weight=0):
        # Construct edge_id using source and target to uniquely identify edges
        edge_id = f"{source}_{target}"
        self.edges[edge_id] = {'source': source, 'target': target, 'weight': weight}

    
    def calculate_correlation(self, node_a, node_b):
        # Filter rows where both node_a and node_b have non-negative values
        filtered_df = self.df[(self.df[node_a] >= 0) & (self.df[node_b] >= 0)]

        # Create binary arrays indicating presence based on the filtered rows
        a = np.array([int(val > 0) for val in filtered_df[node_a]])
        b = np.array([int(val > 0) for val in filtered_df[node_b]])

        # Calculate and return the correlation
        return matthews_corrcoef(a, b)

    
    @staticmethod
    def visualize_graph(ResIN, question_mapping, node_partisan_data, show_edges):
        G = nx.Graph()

        for node in ResIN.nodes:
            label = question_mapping.get(node, node)
            # partisan_score = node_partisan_data.get(node, 4)  # Default to a neutral score if absent
            G.add_node(node, label=label) #, partisan=partisan_score)


        for edge in ResIN.edges.values():
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=20, cmap=plt.cm.coolwarm)# node_color=colors

        node_labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')


        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w for w in weights], edge_color='gray')

        if show_edges:
            edge_labels = {(u, v): f"{w:.2f}" for (u, v, w) in G.edges(data='weight')}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)


        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=1, vmax=10))
        sm.set_array([])
        # plt.colorbar(sm, label="Partisan Leaning")

        plt.title("Response Item Network")
        plt.axis('off')
        plt.show()

