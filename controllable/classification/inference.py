

from transformers import CLIPProcessor ,CLIPModel
import torch 
from spacy import load
import wandb 
 
from controllable.classification.train_verify import freeze, setup_args_for_verify, test_phase, train_loop, verify_init
from controllable.classification.util.data_util import get_data_loader
import os
import shutil 
from controllable.classification.util.enums import ModelAttribute 

def load_model(checkpoint_path,model):
    # path=os.path.join(checkpoint_dir,"base.pt")
    model.load_state_dict(torch.load(checkpoint_path),strict=False)#
    model.eval()
    
    return model 

def copy_check_point(checkpoint_dir,target_dir):
    path=os.path.join( checkpoint_dir,"base.pt")
    target_path=os.path.join(target_dir,"base.pt")
    shutil.copy(path, target_path)
    return target_dir

def inference( config_kwargs):
    # args.checkpoint_dir=copy_check_point(args.checkpoint_dir,args.final_checkpoint_dir)
    # args,logger=setup_args_for_verify(config_kwargs)
    
    train_loop(None,config_kwargs)
     
     