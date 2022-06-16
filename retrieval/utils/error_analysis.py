
import os 
import pandas as pd  
from random import sample

from verification.util.error_analysis import sample_corpus


def gen_verification_error_dir(data_path ):
    return data_path+"/retrieval/error" 
    

def merge_result(data_path,verification_result_dir,verification_error_dir,out_name,number):
    retrieval_corpus=os.path.join(data_path,"Corpus2.csv")
    verification_corpus=os.path.join(verification_result_dir,"retrieval_result.csv")
    out_corpus= os.path.join(verification_error_dir,out_name)
    
    retrieval_corpus_df = pd.read_csv(retrieval_corpus ,encoding="utf8")  
    verification_corpus_df = pd.read_csv(verification_corpus ,encoding="utf8")  
    evidence_df=retrieval_corpus_df.merge(verification_corpus_df, how='inner',   on="claim_id",suffixes=(None,"_predict")) 
    evidence_df=evidence_df.rename(columns={"img_evidences":"image_evidence_predict"})
    # evidence_df=evidence_df[["claim_id",'Claim','Evidence','cleaned_truthfulness', 'predicted_truthfulness']]
    evidence_df =sample_corpus(evidence_df,number)
    evidence_df.to_csv(out_corpus,index=False) 


def retrieval_error_analysis(data_path):
    out_name="retrieval_error.csv"
    verification_dir=os.path.join(data_path,"retrieval")
    verification_error_dir=gen_verification_error_dir(data_path )
    os.makedirs(verification_error_dir, exist_ok=True)
    
    merge_result(data_path,verification_dir,verification_error_dir,out_name,50)
    generate_report_sheet(data_path,out_name) 
    

def generate_report_sheet(data_path,out_name):                                         
    corpus= os.path.join(gen_verification_error_dir(data_path ),out_name)
    corpus_df = pd.read_csv(corpus ,encoding="utf8")  
    corpus_df = corpus_df.drop_duplicates(subset='claim_id', keep="first")
    corpus_df.to_csv(os.path.join(gen_verification_error_dir(data_path ),"retrieval_error_claim_level.csv"),index=False) 