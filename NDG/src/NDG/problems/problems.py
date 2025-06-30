# from machinelearning import *
# from mathematics import *
# from optimization import *
# from physics import *
class Probs():
    def __init__(self,paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            from .optimization.tsp_greedy import run
            self.prob = run.TSPCONST()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "bp_online":
            from .optimization.bp_online import run
            self.prob = run.BPONLINE()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "hicnet":
            from .optimization.hicnet import run
            self.prob = run.HICNET()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "butter":
            from .optimization.butter import run
            self.prob = run.BUTTER()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "cora":
            from .optimization.cora import run
            self.prob = run.CORA()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "dream":
            from .optimization.dream import run
            self.prob = run.DREAM()
            print("- Prob "+paras.problem+" loaded ")
        elif paras.problem == "mc":
            from .optimization.mc import run
            self.prob = run.MC()
            print("- Prob "+paras.problem+" loaded ")
        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):

        return self.prob
