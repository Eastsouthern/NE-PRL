class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "给定带噪声邻接矩阵，您的任务是对网络进行去噪以获得增强的邻接矩阵。"
            "目标是通过减少噪声和改善连接的表示来提高网络的准确性。增强后的网络将被用于节点分类任务。"
            "设计一个创新的算法，该算法能够迭代分析带噪声的邻接矩阵，并根据连接强度调整连接。"
            "该算法应该具有创新性，并与现有文献方法不同。"
            "在提高准确性的同时，确保保持网络的结构特性。"
        )
        self.prompt_func_name = "denoise_network"
        self.prompt_func_inputs = ["noisy_adj"]
        self.prompt_func_outputs = ["denoised_adj"]
        self.prompt_inout_inf = (
            "'noisy_adj'是输入的带噪声邻接矩阵。"
            "'denoised_adj'是输出的去噪后的邻接矩阵。"
        )
        self.prompt_other_inf = (
            "两个矩阵'noisy_adj'和'denoised_adj'都是Numpy数组。"
            "请确保在代码开始时导入所需的包。"
        )

    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf 