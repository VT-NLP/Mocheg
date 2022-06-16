from retrieval.utils import EasyDict

config =  EasyDict()
config.verbose=False


def gen_config_str():
    config_str=""
    for key,value in config.items():
        config_str+=key+":"+str(value)+";  "
    return config_str

def gen_run_desc_from_config():
    run_desc=""
    return run_desc
        