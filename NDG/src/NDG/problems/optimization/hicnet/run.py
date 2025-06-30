import numpy as np
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import community as community_louvain
from .get_instance import GetData
from .prompts import GetPrompts
import warnings
import types
import sys
import re

class HICNET():
    
    def __init__(self):
        self.getData = GetData()
        self.noisy_adjacency_matrix, self.labels = self.getData.load_data()
        self.n_nodes = self.noisy_adjacency_matrix.shape[0]
        self.prompts = GetPrompts()
        
    def evaluate(self, code_string):
        try:
            # 预处理代码字符串
            code_string = re.sub(r'\{[^}]*\}', '', code_string)
            code_lines = [line for line in code_string.split('\n') if line.strip()]
            code_string = '\n'.join(code_lines)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                denoising_module = types.ModuleType("denoising_module")
                
                compile(code_string, '<string>', 'exec')
                exec(code_string, denoising_module.__dict__)
                sys.modules[denoising_module.__name__] = denoising_module
                fitness = self.denoise(denoising_module)
                return fitness
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            return None
        
    def denoise(self, module):
        # Use the loaded noisy adjacency matrix directly
        noisy_adjacency_matrix = self.noisy_adjacency_matrix

        # Ensure the denoising function exists in the module
        if hasattr(module, 'enhance_network'):
            denoised_adj = module.enhance_network(noisy_adjacency_matrix)
            # Calculate the NMI for the original and denoised adjacency matrices
            nmi_raw = self.calculate_MSE(noisy_adjacency_matrix, self.labels)
            nmi_denoised = self.calculate_MSE(denoised_adj, self.labels)
            print(f"The MSE on raw network is {nmi_raw:.4f}")
            print(f"The MSE on enhance_network is {nmi_denoised:.4f}")
            return nmi_denoised
        else:
            raise AttributeError("The provided module does not have a 'enhance_network' function.")

    def calculate_NMI(self, W, labels):
        # Create a graph object from the adjacency matrix
        G = nx.from_numpy_array(W)
        
        # Perform community detection using Louvain method
        partition = community_louvain.best_partition(G, weight='weight')
        predicted_labels = [partition[node] for node in range(self.n_nodes)]  # 使用节点索引作为标识符
        
        # Calculate NMI
        nmi = normalized_mutual_info_score(labels, predicted_labels)
        return 1-nmi
    
    def calculate_MSE(self,W, labels):
        # 计算MSE
        mse = np.mean((W - labels) ** 2)
        return mse