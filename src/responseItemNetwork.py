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

    def build_graph(self):
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node_a, node_b = self.nodes[i], self.nodes[j]
                correlation = self.calculate_correlation(node_a, node_b)
                if correlation > 0:  # Only add positive associations
                    self.add_edge(node_a, node_b, correlation)

    def binarize_df(self, df):
        pass
        """Binarize the dataframe."""
        return df.applymap(lambda x: 1 if x == x else 0)    

    def add_edge(self, edge, source, target, weight=0):
        edge_id = f"{edge['source']}_{edge['target']}"
        self.edges[edge_id] = {'source': source, 'target': target, 'weight': weight}
    
    def calculate_correlation(self, node_a, node_b):
        """Calculate correlation between two binary variables (nodes) using Matthews Correlation Coefficient."""
        a = np.array([int(node_a in response) for response in self.responses.values()])
        b = np.array([int(node_b in response) for response in self.responses.values()])
        return matthews_corrcoef(a, b)
    
    @staticmethod
    def visualize_graph(ResIN):
        G = nx.Graph()

        for node in ResIN.nodes:
            G.add_node(node)

        for edge_id, edge in ResIN.edges.items():
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights], edge_color='gray')

        edge_labels = {(u, v): f"{w:.2f}" for (u, v, w) in G.edges(data='weight')}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Response Item Network")
        plt.axis('off')
        plt.show()

