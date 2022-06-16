import os 
import re
import json
def setup_with_args(args,outdir,run_desc=""):
    if  run_desc !=None and args.desc!=None:
        run_desc+="-"+args.desc
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
    print(f'Creating output directory...{args.run_dir}')
    os.makedirs(args.run_dir)
    
    
    
    # Print options.
    # print()
    # print(f'Training options:  ')
    # print(json.dumps(args, indent=2))
    # print()
    # print(f'Output directory:   {args.run_dir}')
    # with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
    #     json.dump(args, f, indent=2)
 
    return args.run_dir,args 