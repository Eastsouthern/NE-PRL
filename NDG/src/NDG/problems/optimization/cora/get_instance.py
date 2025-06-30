import numpy as np
import pandas as pd
import scipy.sparse as sp

class GetData():
    def __init__(self, 
                 cites_file_path='./cora.cites',
                 content_file_path='./cora.content'):
        self.cites_file_path = cites_file_path
        self.content_file_path = content_file_path

    def load_data(self):
        # 加载节点特征和标签
        content_df = pd.read_csv(self.content_file_path, sep='\t', header=None)
        # 第一列是节点ID，最后一列是标签
        node_map = {node_id: idx for idx, node_id in enumerate(content_df[0])}
        labels = pd.Categorical(content_df.iloc[:, -1]).codes
        
        # 加载边信息并构建邻接矩阵
        cites_df = pd.read_csv(self.cites_file_path, sep='\t', header=None)
        edges = [(node_map[row[0]], node_map[row[1]]) for _, row in cites_df.iterrows()]
        
        n_nodes = len(node_map)
        # 将edges转换为两个数组
        rows, cols = zip(*edges)
        
        # 正确构建coo_matrix
        adj_matrix = sp.coo_matrix(
            (np.ones(len(edges)), (rows, cols)),
            shape=(n_nodes, n_nodes)
        ).toarray()
        
        # 使原始邻接矩阵对称化
        # adj_matrix = adj_matrix + adj_matrix.T
        # adj_matrix = np.clip(adj_matrix, 0, 1)
        
        return adj_matrix, labels 
    
# # 测试加载数据
# if __name__ == "__main__":
#     data = GetData(cites_file_path='netexam\cora\cora.cites', content_file_path='netexam\cora\cora.content')
#     adj_matrix, labels = data.load_data()
#     print(adj_matrix)
#     print(labels)