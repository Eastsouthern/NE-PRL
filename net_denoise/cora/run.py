from NDG import NDG
from NDG.utils.getParas import Paras

# Parameter initialization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "NDG",    # ['ael','NDG']
                problem = "cora", 
                llm_api_endpoint = "api.chatanywhere.tech", # set your LLM endpoint
                llm_api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # set your LLM API key        
                llm_model = "gpt-4o-mini",
                ec_pop_size = 6,  # number of samples in each population
                ec_n_pop = 25,   # number of populations
                exp_n_proc = 8,   # number of parallel processes
                exp_debug_mode = False,
                )

# initialization
evolution = NDG.EVOL(paras)

# run 
evolution.run()
