from NDG import NDG
from NDG.utils.getParas import Paras

# Parameter initialization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "NDG",    # ['ael','NDG']
                problem = "butter", 
                llm_api_endpoint = "api.chatanywhere.tech", # set your LLM endpoint
                llm_api_key = "sk-xxxxxxx", # set your LLM API         
                llm_model = "gpt-4o-mini",
                ec_pop_size = 10,  # number of samples in each population
                ec_n_pop = 4,   # number of populations
                exp_n_proc = 8,   #  number of parallel processes
                exp_debug_mode = False,
                management = 'pop_greedy',  # management strategy
                selection = 'tournament',    # selection strategy
                # evaluation parameters  
                eva_timeout = 120,  # timeout for evaluation in seconds
                eva_numba_decorator = False,  # whether to use numba for evaluation
                )

# initialization
evolution = NDG.EVOL(paras)

# run 
evolution.run()