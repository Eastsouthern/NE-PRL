class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            "Given a noisy adjacency matrix of a network, your task is to denoise "
            "the network to retrieve an enhanced adjacency matrix. The goal is to improve the "
            "accuracy of the network by reducing noise and improving the representation of connections. "
            "Design a novel algorithm that iteratively analyzes the noisy adjacency matrix and adjusts "
            "the connections based on their strengths. The algorithm should be innovative and different "
            "from existing literature methods. Ensure to maintain the network's structural properties while "
            "enhancing its accuracy."
        )
        self.prompt_func_name = "denoise_network"
        self.prompt_func_inputs = ["noisy_adj"]
        self.prompt_func_outputs = ["denoised_adj"]
        self.prompt_inout_inf = (
            "'noisy_adj' is the input noisy adjacency matrix. "
            "'denoised_adj' is the output denoised adjacency matrix."
        )
        self.prompt_other_inf = (
            "Both 'noisy_adj' and 'denoised_adj' are Numpy arrays. "
            "Please make sure to import any necessary packages at the beginning of the code."
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

if __name__ == "__main__":
    getprompts = GetPrompts()
    print(getprompts.get_task())