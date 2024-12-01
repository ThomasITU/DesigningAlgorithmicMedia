import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA

class ResponseItemNetwork:

    def __init__(self, df: pd.DataFrame, question_mapping: dict[str, tuple[str, int]]):
        self.question_mapping = question_mapping # {question key (X99): [question name, range of possible_answers (int)}
        self.df = self.binarize_df(df)
        self.nodes = list(self.df.columns)
        self.edges = {}
        self.build_graph()

    def build_graph(self):
        # Create a unique list of nodes (ensure no duplicates)
        seen_nodes = set()  # To track questions we already processed

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node_a, node_b = self.nodes[i], self.nodes[j]
                
                # Skip nodes that are the same question (using the first part of the name before the '_')
                if node_a.split('_')[0] == node_b.split('_')[0]:
                    continue

                # Skip if we've already processed this pair
                if (node_a, node_b) in seen_nodes or (node_b, node_a) in seen_nodes:
                    continue

                # Track the pair to avoid redundant processing
                seen_nodes.add((node_a, node_b))

                # Calculate correlation between node_a and node_b
                try:
                    correlation = self.calculate_correlation(node_a, node_b)
                    if correlation > 0:  # Only add positive associations from the paper
                        print(f"Adding edge between {node_a} and {node_b} with correlation {correlation}")
                        self.add_edge(node_a, node_b, correlation)
                except ValueError as e:
                    print(f"Error calculating correlation between {node_a} and {node_b}: {e}")



    def binarize_df(self, df):
        # Create an empty DataFrame to store results
        binarized = pd.DataFrame(index=df.index)

        for key, (question, possible_answers) in self.question_mapping.items():
            # Create a single binary row per respondent for all possible answers
            binarized_answers = pd.get_dummies(df[key], prefix=question)
            
            # Add missing possible answers as columns with all zeros (if necessary)
            for answer in range(1, possible_answers + 1):
                column_name = f"{question}_{answer}"
                if column_name not in binarized_answers:
                    binarized_answers[column_name] = 0

            # Concatenate the binarized answers for this question to the result
            binarized = pd.concat([binarized, binarized_answers], axis=1)

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

    
    @staticmethod
    def visualize_graph(ResIN, question_mapping, node_partisan_data, show_edges, weight_multiplier=10):
        pca = PCA(n_components=2)
        G = nx.Graph()

        for node in ResIN.nodes:
            if question_mapping is None or node not in question_mapping:
                G.add_node(node, label=node)
                continue
            label = question_mapping.get(node, node)
            # partisan_score = node_partisan_data.get(node, 4)  # Default to a neutral score if absent
            G.add_node(node, label=label) #, partisan=partisan_score)


        for edge in ResIN.edges.values():
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

        pos = nx.spring_layout(G, iterations=5000)
        positions = ResponseItemNetwork.extract_positions(pos)
        # pca.fit(positions)
        # x_pca = pca.transform(positions)
        # xx = x_pca[:, 0]
        # yy = x_pca[:, 1]

        # mm = min(xx)*1.1
        # MM = max(xx)*1.1
        nx.draw_networkx_nodes(G, pos, node_size=30, cmap=plt.cm.coolwarm)# node_color=colors

        node_labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=5)


        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w/weight_multiplier for w in weights], edge_color='gray')

        if show_edges:
            edge_labels = {(u, v): f"{w:.2f}" for (u, v, w) in G.edges(data='weight')}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)


        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=1, vmax=10))
        sm.set_array([])
        # plt.colorbar(sm, label="Partisan Leaning")

        plt.title("Response Item Network")
        # plt.axis('off')
        # plt.xlim([mm, MM])
        # plt.ylim([mm, MM])

        plt.show()

    @staticmethod
    def extract_positions(pos):
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