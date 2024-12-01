import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

class ResponseItemNetwork:

    CMAP = plt.get_cmap("seismic")
    POLITCAL_BELIEF_COLUMN = 'resin_political_beliefs'
    def __init__(self, df: pd.DataFrame, question_mapping: dict[str, tuple[str, int, bool]], political_belief_column: str | None):
        self.question_mapping = question_mapping # {question key (X99): [question name, range of possible_answers (int), is_inverted}
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

        for key, (question, belief_spectrum, is_inverted) in self.question_mapping.items(): 
            if is_inverted:
                df[key] = self.invert_question_mapping(belief_spectrum, df[key])
            for answer in range(1, belief_spectrum + 1): # iterate over the possible range of answers to a question
                column_name = f"{question}_{answer}" # Create a column name for the belief e.g. belief_1, belief_2, etc 
                binarized[column_name] = (df[key] == answer).astype(int) # Add a 1 if the belief is held at the "answer" level , otherwise 0
       
        return binarized
    
    def invert_question_mapping(self, max_scale, column_data):
        column_data.apply(lambda x: (max_scale + 1) - x)

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
        corr, _ = pearsonr(a, b)
        return  corr 
    
    def linearization_score(self, positions):
        # Calculate the linearization score for the network based on the paper
        x_positions = [coord[0] for coord in positions.values()]
        y_positions = [coord[1] for coord in positions.values()]

        linearization_score = (max(x_positions) - min(x_positions)) / (max(y_positions) - min(y_positions))
        return linearization_score 

    def visualize_graph(self, show_node_labels=True):
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
        print(f"Linearization Score Before rotation: {self.linearization_score(positions)}")

        # rotate positions to align political leaning left -> right
        if sum([G.nodes[node]["leaning"] for node in G.nodes]) > 0:
            positions = self.lock_alignment(positions, G)
        
        # draw nodes
        plt.figure(figsize=(12, 7))  # Adjust the figure size (width, height)
        
        node_labels = {node: data.get('label', node) for node, data in G.nodes(data=True)} if show_node_labels else {}
        nx.draw_networkx_nodes(G, positions, node_size=500, 
                                node_color=node_colors, edgecolors='lightgray', linewidths=2)
        nx.draw_networkx_labels(G, positions, labels=node_labels, font_size=8)
        
        # draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, positions, width=[w for w in weights], edge_color='gray')

        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=self.CMAP), ax=plt.gca())
        cbar.set_label('Mean Political Belief', fontsize=12)  

        # Remove the color bar ticks (legend text) and adjust the font size
        cbar.set_ticks([])  # Remove the ticks
        cbar.ax.tick_params(labelsize=0)  # Hide the tick labels completely
        cbar.ax.yaxis.set_ticks_position('none')  # Remove ticks from the color bar

        plt.title("Response Item Network")
        plt.show()

    # 1 = left, 10 = right
    def get_mean_political_belief(self, feature_node, political_beliefs: pd.Series):
        opinion = self.df[feature_node]
        mean_belief = np.mean(political_beliefs[opinion > 0])
        return mean_belief

   # Map each question's political belief mean to a color,                                                                                        
    def political_belief_to_color(self, mean_belief, belief_scale=9):
        # Normalize the mean political belief to [0, 1] to create a color gradient
        norm_belief = (mean_belief - 1) / belief_scale  # Normalize to [0, 1] (1 = left, 10 = right)
        return self.CMAP(norm_belief)  # Map the normalized value to the color


    def lock_alignment(self, pos, G: nx.Graph):
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
    
    def varimax_rotation(self, pos, G: nx.Graph):
        # based on the original Respondent network 
        scaler = StandardScaler()
        positions = np.array([list(pos[node]) for node in G.nodes]) 

        positions_scaled = scaler.fit_transform(positions)
        pca = PCA(n_components=2)
        postions_pca = pca.fit_transform(positions_scaled)

        angle = np.arctan2(pca.components_[1, 0], pca.components_[1, 1])

        # Rotation matrix (2D rotation matrix to align first principal component with the X-axis)
        rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
                                    [np.sin(-angle), np.cos(-angle)]])
        
        # Apply the rotation to the PCA coordinates
        rotated_coords = np.dot(postions_pca, rotation_matrix)

        final_coords = scaler.inverse_transform(rotated_coords)

        # Ensure "left-leaning" nodes are on the left side of the plot
        leaning_values = [G.nodes[node]["leaning"] for node in G.nodes]
        left_indices = [i for i, val in enumerate(leaning_values) if val < 5]
        right_indices = [i for i, val in enumerate(leaning_values) if val > 5]        

        x_coords = final_coords[:, 0]
        y_coords = final_coords[:, 1]

        # Ensure that left-leaning nodes are on the left side of the plot
        avg_left_x = np.mean([x_coords[i] for i in left_indices])
        avg_right_x = np.mean([x_coords[i] for i in right_indices])

        avg_left_y = np.mean([y_coords[i] for i in left_indices])
        avg_right_y = np.mean([y_coords[i] for i in right_indices])

        # Flip x-axis if necessary
        if avg_left_x > avg_right_x:
            final_coords[:, 0] = -final_coords[:, 0]

        # Flip y-axis if necessary 
        if avg_left_y < avg_right_y:
            final_coords[:, 1] = -final_coords[:, 1]  

        # Step 5: Map the new coordinates back to the original graph
        adjusted_positions = {node: final_coords[i] for i, node in enumerate(G.nodes())}
        return adjusted_positions