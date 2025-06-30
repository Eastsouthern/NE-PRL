import numpy as np

class GetData():
    
    def __init__(self, file_path='./few_shot'):
        # self.label_file_path =  './net_label/5K/class_10_5000.txt'
        # self.mat_file_path = './net_label/5K/mat_10_5000.txt'
        
        self.label_file_path =  './data_02/network_original.txt'
        self.mat_file_path = './data_02/network_noisy.txt'
        
    def load_data(self):
        # with open(self.label_file_path, 'r') as file:
        #     labels = file.read().split(',')
        #     labels = [int(label) for label in labels]
        
        # adjacency_matrix = []
        # with open(self.mat_file_path, 'r') as file:
        #     lines = file.readlines()
        #     for line in lines:
        #         row = [float(num) for num in line.split(',')]
        #         adjacency_matrix.append(row)
        
        # adjacency_matrix = np.array(adjacency_matrix)
        
        adjacency_matrix = np.loadtxt(self.mat_file_path, delimiter=',')
        labels = np.loadtxt(self.label_file_path, delimiter=',')
        
        return adjacency_matrix, labels
