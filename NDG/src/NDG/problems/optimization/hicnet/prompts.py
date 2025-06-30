class GetPrompts():
    def __init__(self):
        self.prompt_task = (
            # Original prompt:
            "Given a noisy adjacency matrix of a network, your task is to denoise the network to retrieve "
            "an enhanced adjacency matrix. In network denoising, some nodes are directly related, while "
            "others are not. However, indirect relationships (e.g., A relates to B, B relates to C, C relates "
            "to D) may create artificial connections (like between A and D) or amplify existing weak "
            "connections beyond their true strength. The goal is to weaken these extra connections while "
            "strengthening the network's true structure. Design a novel algorithm that iteratively analyzes "
            "the noisy adjacency matrix and adjusts the connections based on their strengths, distinguishing "
            "between direct and indirect relationships. The algorithm should be innovative and different from "
            "existing literature methods. Ensure to maintain the network's core structural properties while "
            "enhancing its accuracy by reducing spurious connections and reinforcing genuine ones."

            # Enhanced prompt with clear role, context, task breakdown, constraints and evaluation criteria:
            # "You are an expert algorithm designer specializing in network analysis and denoising. "
            # "Context: You are working with a noisy adjacency matrix representing network connections. The matrix contains "
            # "both true connections and spurious ones caused by indirect relationships.\n\n"
            # "Task: Create an innovative algorithm to denoise this network by:\n"
            # "1. Analyzing connection patterns to identify direct vs indirect relationships\n"
            # "2. Developing a method to quantify connection strengths\n"
            # "3. Implementing an iterative process to adjust these connections\n\n"
            # "Constraints:\n"
            # "- The algorithm must be novel and not replicate existing methods\n"
            # "- Must preserve the network's fundamental structure\n"
            # "- Should handle both weakening false connections and strengthening true ones\n\n"
            # "Success Criteria:\n"
            # "- Effectively distinguishes between direct and indirect relationships\n"
            # "- Maintains network structural integrity\n"
            # "- Produces a cleaner, more accurate representation of the true network\n\n"
            # "Output: Provide a Python function that implements this denoising algorithm, with detailed comments "
            # "explaining your approach and methodology."
        )
        self.prompt_func_name = "enhance_network"
        self.prompt_func_inputs = ["noisy_adj"]
        self.prompt_func_outputs = ["enhanced_adj"]
        self.prompt_inout_inf = (
            "'noisy_adj' is the input noisy adjacency matrix. "
            "'enhanced_adj' is the output adjacency matrix with enhanced community structure."
        )
        self.prompt_other_inf = (
            "Both 'noisy_adj' and 'enhanced_adj' are Numpy arrays. "
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