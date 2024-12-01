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
        seen_nodes = set()

        for i, node_a in enumerate(self.nodes):
            for node_b in self.nodes[i + 1:]:
                # Skip nodes we don't want to process multiple times
                if self.skip_seen_nodes(seen_nodes, node_a, node_b):
                    continue

                # Calculate correlation and add edge if positive based on the paper
                correlation = self.calculate_correlation(node_a, node_b)
                if correlation > 0:
                    self.add_edge(node_a, node_b, correlation)

    # helper method to skip nodes that we don't want to process multiple times
    def skip_seen_nodes(self, seen_nodes: set, node_a: str, node_b: str):
            # Skip political belief columns and nodes that are the same question
            if node_a == self.POLITCAL_BELIEF_COLUMN or node_b == self.POLITCAL_BELIEF_COLUMN or node_a.split('_')[0] == node_b.split('_')[0]:
                return True
            
            # Skip if we've already processed this pair
            if (node_a, node_b) in seen_nodes or (node_b, node_a) in seen_nodes:
                return True
            
            seen_nodes.add((node_a, node_b)) 
            return False

    def binarize_df(self, df, political_belief_column):
        binarized = pd.DataFrame(index=df.index)
        if political_belief_column:
            binarized[self.POLITCAL_BELIEF_COLUMN] = df[political_belief_column] # Add political beliefs to the binarized dataframe

        for key, (question, belief_spectrum) in self.question_mapping.items(): 
            for answer in range(1, belief_spectrum + 1): # iterate over the possible range of answers to a question
                column_name = f"{question}_{answer}" # Create a column name for the belief e.g. belief_1, belief_2, etc 
                binarized[column_name] = (df[key] == answer).astype(int) # Add a 1 if the belief is held at the "answer" level , otherwise 0
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

        return matthews_corrcoef(a, b) # Calculate and return the correlation
        

    def visualize_graph(self, weight_multiplier=10):
        G = nx.Graph()
        
        # Create a list of colors for each node based on the mean political belief
        node_colors = []
        partisan_column = self.df[self.POLITCAL_BELIEF_COLUMN] if (self.POLITCAL_BELIEF_COLUMN in self.df.columns) else pd.Series()

        for node in self.nodes:
            if node == self.POLITCAL_BELIEF_COLUMN:
                continue

            # Calculate the political mean belief in order to color each question node
            mean_belief = 0
            if not partisan_column.empty:
                mean_belief = self.get_mean_political_belief(node, partisan_column)
                node_colors.append(self.political_belief_to_color(mean_belief))
            G.add_node(node, label=node, leaning=mean_belief)


        # Add edges to the graph
        for edge in self.edges.values():
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

        # Calculate the positions of the nodes using force-directed layout
        positions = nx.spring_layout(G, iterations=5000)

        # rotate positions to align political leaning left -> right
        if sum([G.nodes[node]["leaning"] for node in G.nodes]) > 0:
            positions = self.extract_positions(positions, G)

        # draw nodes
        node_labels = {node: data.get('label', node) for node, data in G.nodes(data=True)}
        nx.draw_networkx_nodes(G, positions, node_size=300, node_color=node_colors)
        nx.draw_networkx_labels(G, positions, labels=node_labels, font_size=8)

        # draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, positions, width=[w * weight_multiplier for w in weights], edge_color='gray')

        plt.title("Response Item Network")
        plt.show()

    # 1 = left, 10 = right
    def get_mean_political_belief(self, feature_node, political_beliefs: pd.Series):
        opinion = self.df[feature_node]
        mean_belief = np.mean(political_beliefs[opinion > 0])  # Avoid division by 0 for missing answers
        return mean_belief

   # Map each question's political belief mean to a color,                                                                                        
    def political_belief_to_color(self, mean_belief, belief_scale=9):
        # Normalize the mean political belief to [0, 1] to create a color gradient
        custom_cmap = plt.get_cmap("seismic")
        norm_belief = (mean_belief - 1) / belief_scale  # Normalize to [0, 1] (1 = left, 10 = right)
        return custom_cmap(norm_belief)  # Map the normalized value to the color


    def extract_positions(self, pos, G: nx.Graph):
        # based on the original Respondent network 
        # https://github.com/just-a-normal-dino/AS22_analysis_RESIN/blob/main/graded_model_full.ipynb
        positions = np.array([list(pos[node]) for node in G.nodes]) 
        pca = PCA(n_components=2)
        rotated_positions = pca.fit_transform(positions)

        # Ensure "left-leaning" nodes are on the left side of the plot
        leaning_values = [G.nodes[node]["leaning"] for node in G.nodes]
        left_indices = [i for i, val in enumerate(leaning_values) if val < 5]
        right_indices = [i for i, val in enumerate(leaning_values) if val > 5]        

        # Extract x-coordinates and node leaning
        x_coords = rotated_positions[:, 0]
        y_coords = rotated_positions[:, 1]

        # Check average x-coordinates
        avg_left_x = np.mean([x_coords[i] for i in left_indices])
        avg_right_x = np.mean([x_coords[i] for i in right_indices])

        # Calculate average y-coordinates for left and right nodes
        avg_left_y = np.mean([y_coords[i] for i in left_indices])
        avg_right_y = np.mean([y_coords[i] for i in right_indices])

        # Flip x-axis if necessary
        if avg_left_x > avg_right_x:
            rotated_positions[:, 0] = -rotated_positions[:, 0]

        # Flip y-axis if necessary try to lock the y-axis such that it does not flip
        if avg_left_y < avg_right_y:
            rotated_positions[:, 1] = -rotated_positions[:, 1]  

        rotated_pos = {node: rotated_positions[i] for i, node in enumerate(G.nodes)}
        return rotated_pos