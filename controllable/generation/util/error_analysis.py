


import os 
import pandas as pd 
from generation.util.preprocess_for_inference import int_truthfulness_to_str ,rename
from random import sample

from verification.util.error_analysis import sample_corpus


def gen_verification_error_dir(data_path ):
    return data_path+"/generation/error" 


def merge_generation(data_path,verification_dir,verification_error_dir,out_name):
    generation_file="/home/menglong/workspace/code/misinformation_detection/controllable/generation/output/bart/run_3/without/generated_predictions.txt"
    reference_file=os.path.join(data_path,"Corpus2_for_controllable_generation_200.csv")
    out_corpus=os.path.join(verification_error_dir,out_name)
    reference_file_df = pd.read_csv(reference_file ,encoding="utf8")  
    with open(generation_file, 'r') as f1_file :
        prediction_list = f1_file.readlines()
        reference_file_df.insert(5, "predicted_explanation",prediction_list )
        reference_file_df =reference_file_df.sample(n= 50)
        reference_file_df=reference_file_df.apply(extract_claim,axis=1)
        reference_file_df.to_csv(out_corpus,index=False)
        print(reference_file_df.columns)
        print(len(reference_file_df) )
        

def extract_claim(series)        :
    truth_claim_evidence=series["truth_claim_evidence"]
    claim=truth_claim_evidence.split("</s>")[1]
    series["claim"]=claim 
    return series
            
def generation_error_analysis(data_path):
    out_name="generation_error.csv"
    verification_dir=os.path.join(data_path,"generation")
    verification_error_dir=gen_verification_error_dir(data_path )
    merge_generation(data_path,verification_dir,verification_error_dir,out_name)
   