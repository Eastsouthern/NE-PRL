from NDG import NDG
from NDG.utils.getParas import Paras

# Parameter initialization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "NDG",    # ['ael','NDG']
                problem = "hicnet", 
                llm_api_endpoint = "api.chatanywhere.tech", # set your LLM endpoint
                llm_api_key = "sk-cOqgMkQ605Om9Z28q3UtAfOAtOn7c53tld94Cu01oOEoVTmg",              
                llm_model = "gpt-4o-mini",
                ec_pop_size = 6,  # number of samples in each population
                ec_n_pop = 10,   # number of populations
                exp_n_proc = 16,   # multi-core parallel
                exp_debug_mode = False)

# initialization
evolution = NDG.EVOL(paras)

# run 
evolution.run()