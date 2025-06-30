from NDG import NDG
from NDG.utils.getParas import Paras

# Parameter initialization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "NDG",    # ['ael','NDG']
                problem = "hicnet", 
                llm_api_endpoint = "api.chatanywhere.tech",
                llm_api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # set your LLM API key
                llm_model = "gpt-4o-mini-ca",
                ec_pop_size = 8,  # number of samples in each population
                ec_n_pop = 10,   # number of populations
                exp_n_proc = 10,   # multi-core parallel
                exp_debug_mode = False)

# initialization
evolution = NDG.EVOL(paras)

# run 
evolution.run()