import numpy as np
import scipy.io
import types
import warnings
import sys
from .get_instance import GetData
from .prompts import GetPrompts

class BUTTER():
    def __init__(self):
        self.getData = GetData()
        self.noisy_adjacency_matrix, self.labels = self.getData.load_data()
        self.n_nodes = self.noisy_adjacency_matrix.shape[0]
        self.prompts = GetPrompts()

    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Create a new module object
                denoising_module = types.ModuleType("denoising_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, denoising_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[denoising_module.__name__] = denoising_module

                # Now you can use the module as you would any other
                fitness = self.denoise(denoising_module)
                return fitness
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            return None

    def denoise(self, module):
        # Use the loaded noisy adjacency matrix directly
        noisy_adjacency_matrix = self.noisy_adjacency_matrix

        # Ensure the denoising function exists in the module
        if hasattr(module, 'denoise_network'):
            denoised_adj = module.denoise_network(noisy_adjacency_matrix)
            # Calculate the accuracy between the noisy and denoised adjacency matrices
            acc_raw = self.calculate_accuracy(noisy_adjacency_matrix, self.labels)
            acc_denoised = self.calculate_accuracy(denoised_adj, self.labels)
            # print(f"The accuracy on raw network is {acc_raw:.4f}")
            print(f"The accuracy on enhanced network is {acc_denoised:.4f}")
            return acc_denoised
        else:
            raise AttributeError("The provided module does not have a 'denoise_network' function.")

    def calculate_accuracy(self, W, labels):
        W = W - np.diag(np.diag(W))  # 去除自连接
        unique_labels = np.unique(labels)

        if labels.ndim == 1:
            labels = labels[:, np.newaxis]

        indexx = []
        leni = []
        
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            indexx.append(indices)
            leni.append(len(indices) - 1)

        sorted_indices = np.argsort(-W, axis=1)  # 对连接强度进行降序排序

        retrieval_acc = []

        for i in range(len(labels)):
            li = labels[i]
            li = np.where(unique_labels == li)[0][0]
            if leni[li] <= 0:
                continue
            retrieval_acc.append(len(np.intersect1d(indexx[li], sorted_indices[i, :leni[li]])) / leni[li])

        mean_acc = np.mean(retrieval_acc) if retrieval_acc else 0
        return 1-mean_acc
    
# if __name__ == "__main__":
#     # 示例代码字符串
#     code_string = "import numpy as np\n\ndef denoise_network(noisy_adj):\n    denoised_adj = np.copy(noisy_adj)\n    n = noisy_adj.shape[0]\n\n    # Assigning weights to edges based on sum of connections\n    weights = np.zeros((n, n))\n    for i in range(n):\n        for j in range(n):\n            if i != j:\n                weights[i, j] = noisy_adj[i, j] + noisy_adj[j, i]\n\n    # Constructing denoised adjacency matrix using weights\n    threshold = np.median(weights)\n    for i in range(n):\n        for j in range(n):\n            if i != j:\n                denoised_adj[i, j] = 1 if weights[i, j] > threshold else 0\n\n    return denoised_adj"
    
#     # 创建NETDENOISE对象并进行评价
#     net_denoise = NETDENOISE()
#     fitness = net_denoise.evaluate(code_string)
#     print(f"增强网络的准确率: {fitness}")