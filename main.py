 
from retrieval.utils.util import EasyDict, remove_img_evidence
import click
import os
import click
import re
import json
import tempfile
import torch
import retrieval.utils as utils
 
from functools import partial
import numpy as np 
from controllable.generation.util.preprocess_for_inference import fix_retrieval_result_after_update_corpus2, preprocess_for_generation_inference

from verification.verification_preprocess import preprocess_for_verification, preprocess_for_verification_one_subset
from controllable.generation.util.generation_preprocess import preprocess_for_generation ,preprocess_for_generation_one_subset

def setup_training_loop_kwargs(config_kwargs):
    args=EasyDict()
    for key,value in config_kwargs.items():
        setattr(args,key,value)
 
    run_desc=""
    
    return run_desc,args
    


class UserError(Exception):
    pass
#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')


    
def setup(config_kwargs,has_log=True):
     
 
    # Setup training options.
    run_desc, args = setup_training_loop_kwargs(config_kwargs)

    outdir=args.outdir
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    args.cur_run_id=cur_run_id
    assert not os.path.exists(args.run_dir)
    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    
    
    
    # Print options.
    print()
    print(f'Training options:  ')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    if has_log:
        # Launch processes.
        print('Launching processes...')
        logger=utils.util.Logger(file_mode='a', should_flush=False) #file_name=os.path.join(args.run_dir, 'log.txt'), 
        logger.write(text=json.dumps(args, indent=2))
    else:
        logger=None
    return args,logger


        
 
def preprocess():
    args = parser_args()
    preprocess_for_verification(args.data_path) 
    preprocess_for_generation(args.data_path) 
 
 
  

import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="data")
    parser.add_argument('--in_dir', help='input',  type=str,default="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/train")
    parser.add_argument('--mode',type=str,help=" ",default="preprocess_for_generation")
    args = parser.parse_args()
    return args  
 
if __name__ == '__main__':
    # main()
    args = parser_args() 
    
    if args.mode=="preprocess_for_generation":
        preprocess_for_generation(args.data_path,"Corpus2.csv",out_name="Corpus2_for_controllable_generation.csv")
    elif args.mode=="preprocess_for_generation_inference":        
        preprocess_for_generation_inference( args.data_path+ "/test")
    elif args.mode=="preprocess_for_verification" :
        preprocess_for_verification( args.data_path)
         