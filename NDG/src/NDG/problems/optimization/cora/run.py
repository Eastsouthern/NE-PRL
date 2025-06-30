import numpy as np
import types
import warnings
import sys
from .get_instance import GetData
from .prompts import GetPrompts
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

class CORA():
    def __init__(self):
        self.getData = GetData()
        self.noisy_adjacency_matrix, self.labels = self.getData.load_data()
        self.n_nodes = self.noisy_adjacency_matrix.shape[0]
        self.prompts = GetPrompts()

    def evaluate(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                denoising_module = types.ModuleType("denoising_module")
                exec(code_string, denoising_module.__dict__)
                sys.modules[denoising_module.__name__] = denoising_module
                fitness = self.denoise(denoising_module)
                return fitness
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            return None

    def denoise(self, module):
        noisy_adjacency_matrix = self.noisy_adjacency_matrix

        if hasattr(module, 'denoise_network'):
            denoised_adj = module.denoise_network(noisy_adjacency_matrix)
            acc_raw = self.calculate_accuracy(noisy_adjacency_matrix, self.labels)
            acc_denoised = self.calculate_accuracy(denoised_adj, self.labels)
            print(f"原始网络的准确率: {acc_raw:.4f}")
            print(f"增强网络的准确率: {acc_denoised:.4f}")
            return acc_denoised
        else:
            raise AttributeError("提供的模块中没有'denoise_network'函数。")

    def calculate_accuracy(self, W, labels, n_clusters=7):
        """使用KMeans聚类计算分类准确率
        
        Args:
            W: 邻接矩阵
            labels: 真实标签
            n_clusters: 聚类数量，默认为7（Cora数据集的类别数）
        
        Returns:
            float: 1减去调整兰德指数(为了与优化问题保持一致，越小越好)
        """
        # 移除自连接
        W = W - np.diag(np.diag(W))
        
        # 使用kmeans进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(W)
        
        # 计算聚类结果与真实标签的ARI分数
        acc = adjusted_rand_score(labels, cluster_labels)
        
        # 返回1-acc使得越小越好，与优化问题保持一致
        return 1 - acc 