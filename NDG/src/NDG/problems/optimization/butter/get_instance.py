# import numpy as np

# class GetData():
#     def __init__(self, n_instance, n_nodes):
#         self.n_instance = n_instance
#         self.n_nodes = n_nodes

#     def generate_instances(self):
#         np.random.seed(2024)
#         instance_data = []
#         for _ in range(self.n_instance):
#             # 生成真实的邻接矩阵
#             adjacency_matrix = np.random.randint(0, 2, (self.n_nodes, self.n_nodes))
#             adjacency_matrix = np.triu(adjacency_matrix, 1)
#             adjacency_matrix += adjacency_matrix.T
#             np.fill_diagonal(adjacency_matrix, 0)
            
#             # 生成带噪声的邻接矩阵
#             noise = np.random.normal(0, 0.4, adjacency_matrix.shape)
#             noisy_adjacency_matrix = adjacency_matrix + noise
#             # 确保带噪声的邻接矩阵在 [0, 1] 范围内
#             noisy_adjacency_matrix = np.clip(noisy_adjacency_matrix, 0, 1)
#             instance_data.append((adjacency_matrix, noisy_adjacency_matrix))
#         return instance_data

import scipy.io

class GetData():
    def __init__(self, mat_file_path = './Raw_butterfly_network.mat'):
        self.mat_file_path = mat_file_path

    def load_data(self):
        mat_contents = scipy.io.loadmat(self.mat_file_path)
        # 假设MAT文件中包含W_butterfly0和labels变量
        adjacency_matrix = mat_contents['W_butterfly0']
        labels = mat_contents['labels'].flatten()
        return adjacency_matrix, labels