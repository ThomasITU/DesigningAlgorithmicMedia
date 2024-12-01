import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA

class ResponseItemNetwork:

    POLITCAL_BELIEF_COLUMN = 'resin_political_beliefs'
    def __init__(self, df: pd.DataFrame, question_mapping: dict[str, tuple[str, int]], political_belief_column: str | None):
        self.question_mapping = question_mapping # {question key (X99): [question name, range of possible_answers (int)}
        self.df = self.binarize_df(df, political_belief_column)
        self.nodes = list(self.df.columns)
        self.edges = {}
        self.build_graph()

    def build_graph(self):
        # Create a unique list of nodes (ensure no duplicates)
        seen_nodes = set()  # To track questions we already processed

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node_a, node_b = self.nodes[i], self.nodes[j]
                
                # Skip nodes that we don't want to process multiple times adds side effects
                if self.skip_seen_nodes(seen_nodes, node_a, node_b): 
                    continue

                # Calculate correlation between node_a and node_b
                correlation = self.calculate_correlation(node_a, node_b)
                if correlation > 0:  # Only add positive associations from the paper
                    self.add_edge(node_a, node_b, correlation)

    # helper method to skip nodes that we don't want to process multiple times
    def skip_seen_nodes(self, seen_nodes: set, node_a: str, node_b: str):
            seen = False

            # Skip political belief columns
            if node_a == self.POLITCAL_BELIEF_COLUMN or node_b == self.POLITCAL_BELIEF_COLUMN:
                seen = True

            # Skip nodes that are the same question (using the first part of the name before the '_')
            if node_a.split('_')[0] == node_b.split('_')[0]:
                seen = True

            # Skip if we've already processed this pair
            if (node_a, node_b) in seen_nodes or (node_b, node_a) in seen_nodes:
                seen = True
            

            if not seen:
                seen_nodes.add((node_a, node_b)) 
            return seen

    def binarize_df(self, df, political_belief_column):
        # Create an empty DataFrame to store results
        binarized = pd.DataFrame(index=df.index)
        if political_belief_column:
            binarized[self.POLITCAL_BELIEF_COLUMN] = df[political_belief_column]

        for key, (question, possible_answers) in self.question_mapping.items():
            for answer in range(1, possible_answers + 1):
                column_name = f"{question}_{answer}"
                binarized[column_name] = (df[key] == answer).astype(int)

        return binarized

    def add_edge(self, source, target, weight=0):
        # Construct edge_id using source and target to uniquely identify edges
        edge_id = f"{source}_{target}"
        self.edges[edge_id] = {'source': source, 'target': target, 'weight': weight}

    def calculate_correlation(self, node_a, node_b):
        # Filter rows where both node_a and node_b have non-negative values
        filtered_df = self.df[(self.df[node_a] >= 0) & (self.df[node_b] >= 0)]
        if len(filtered_df) == 0:
            return 0
        
        # Create binary arrays indicating presence based on the filtered rows
        a = np.array([int(val > 0) for val in filtered_df[node_a]])
        b = np.array([int(val > 0) for val in filtered_df[node_b]])

        # Calculate and return the correlation
        return matthews_corrcoef(a, b)

    def visualize_graph(self, weight_multiplier=10):
        G = nx.Graph()
        
        # Create a list of colors for each node based on the mean political belief
        node_colors = []
        partisan_column = self.df[self.POLITCAL_BELIEF_COLUMN] if (self.POLITCAL_BELIEF_COLUMN in self.df.columns) else pd.Series()

        for node in self.nodes:
            if node == self.POLITCAL_BELIEF_COLUMN:
                continue

            # Add the node to the graph
            G.add_node(node, label=node)

            # Calculate the political mean belief in order to color each question node
            if not partisan_column.empty:
                mean_belief = self.get_mean_political_belief(node, partisan_column)
                node_colors.append(self.political_belief_to_color(mean_belief))

        # Add edges to the graph
        for edge in self.edges.values():
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

        # Calculate the positions of the nodes using force-directed layout
        pos = nx.spring_layout(G, iterations=5000)

        # rotate and scale the positions
        # positions = self.extract_positions(pos)
        # pca = PCA(n_components=2)
        # pca.fit(positions)
        # x_pca = pca.transform(positions)
        # xx = x_pca[:, 0]
        # yy = x_pca[:, 1]

        # mm = min(xx)*1.1
        # MM = max(xx)*1.1

        # draw nodes
        node_labels = {node: data.get('label', node) for node, data in G.nodes(data=True)}
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

        # draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w * weight_multiplier for w in weights], edge_color='gray')

        plt.title("Response Item Network")
        # plt.axis('off')
        # plt.xlim([mm, MM])
        # plt.ylim([mm, MM])
        plt.show()

    def get_mean_political_belief(self, feature_node, political_beliefs: pd.Series):
        opinion = self.df[feature_node]
        mean_belief = np.mean(political_beliefs[opinion > 0])  # Avoid division by 0 for missing answers
        return mean_belief

   # Map each question's political belief mean to a color
    def political_belief_to_color(self, mean_belief, belief_scale=9):
        # Normalize the mean political belief to [0, 1] to create a color gradient
        norm_belief = (mean_belief - 1) / belief_scale  # Normalize to [0, 1] (1 = left, 10 = right)
        norm_belief = np.clip(norm_belief, 0, 1)

        # Use a colormap (coolwarm from blue to red) to create the final color
        cmap = plt.get_cmap("coolwarm")
        return cmap(norm_belief)  # Map the normalized value to the color


    def extract_positions(self, pos):
        # based on the original Respondent network 
        # https://github.com/just-a-normal-dino/AS22_analysis_RESIN/blob/main/graded_model_full.ipynb
        pos2 = [[],[]]
        key_list = []
        for key in pos:
            pos2[0].append(pos[key][0])
            pos2[1].append(pos[key][1])
            key_list.append(key)

        pos3 = []
        for key in pos:
            pos3.append([pos[key][0],pos[key][1]])

        return pos3