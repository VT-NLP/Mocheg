import os 
import pandas as pd 
from generation.util.preprocess_for_inference import int_truthfulness_to_str ,rename
from random import sample


def gen_verification_error_dir(data_path ):
    return data_path+"/verification/error" 
    

def save_verification(data_path,verification_result_dir,verification_error_dir,out_name):
    retrieval_corpus=os.path.join(data_path,"Corpus2.csv")
    verification_corpus=os.path.join(verification_result_dir,"verification_result.csv")
    out_corpus= os.path.join(verification_error_dir,out_name)
    
    retrieval_corpus_df = pd.read_csv(retrieval_corpus ,encoding="utf8")  
    verification_corpus_df = pd.read_csv(verification_corpus ,encoding="utf8")  
    verification_corpus_df=int_truthfulness_to_str(verification_corpus_df)
    evidence_df=retrieval_corpus_df.merge(verification_corpus_df, how='inner',   on="claim_id",suffixes=(None,"_y")) 
    evidence_df=evidence_df.rename(columns={"cleaned_truthfulness_y":"predicted_truthfulness"})
    # evidence_df=evidence_df[["claim_id",'Claim','Evidence','cleaned_truthfulness', 'predicted_truthfulness']]
    evidence_df.to_csv(out_corpus,index=False) 

def extract_image_error(corpus_df,data_path):
    _,claim_id_list_with_image_evidence=gen_claim_id_list_with_image_evidence(data_path)
    corpus_df=corpus_df[corpus_df["claim_id"].isin(claim_id_list_with_image_evidence)]
    return corpus_df
    
def gen_claim_id_list_with_image_evidence(data_path)    :
    img_claim_set= set()
    img_evidence_claim_set=set()
    image_corpus=os.path.join(data_path,"images")
    img_list=os.listdir(image_corpus)
    for img_name in img_list:
        prefix=img_name[:17]
        ids=prefix.split("-")
        claim_id= int(ids[0]) 
        relevant_document_id=ids[1]
        img_claim_set.add(claim_id)
        if relevant_document_id=="proof":
            img_evidence_claim_set.add(claim_id)
    img_claim_id_list=list(img_claim_set)
    img_evidence_claim_list=list(img_evidence_claim_set)
    return  img_claim_id_list,img_evidence_claim_list
     
        
    
def extract_error(data_path,verification_error_dir,out_name,number):
    corpus= os.path.join(verification_error_dir,out_name)
 
    corpus_df = pd.read_csv(corpus ,encoding="utf8")  
    corpus_df=corpus_df[corpus_df['cleaned_truthfulness'] != corpus_df['predicted_truthfulness'] ]
    corpus_df=corpus_df[corpus_df['cleaned_truthfulness'] != "NEI"]
    corpus_df=extract_image_error(corpus_df,data_path)
    if number is not None:
        corpus_df =sample_corpus(corpus_df,number)
    corpus_df.to_csv(corpus,index=False) 
    
def sample_corpus(corpus_df,number):
    total_claim_id_list=list(set(corpus_df['claim_id'].tolist()))
    if number<len(total_claim_id_list):
        sampled_claim_id_list=sample(total_claim_id_list,number)
        corpus_df=corpus_df[corpus_df['claim_id'].isin(sampled_claim_id_list)] 
    
    
    return corpus_df 
    
    
def generate_report_sheet(data_path,out_name):                                         
    corpus= os.path.join(gen_verification_error_dir(data_path ),out_name)
    corpus_df = pd.read_csv(corpus ,encoding="utf8")  
    corpus_df = corpus_df.drop_duplicates(subset='claim_id', keep="first")
    corpus_df.to_csv(os.path.join(gen_verification_error_dir(data_path ),"verification_error_claim_level.csv"),index=False) 
    
    

def error_analysis(data_path):
    out_name="verification_error.csv"
    verification_dir=os.path.join(data_path,"verification")
    verification_error_dir=gen_verification_error_dir(data_path )
    save_verification(data_path,verification_dir,verification_error_dir,out_name)
    extract_error( data_path,verification_error_dir,out_name,500)
    generate_report_sheet(data_path,out_name) 
    
    
#verification_result_top_dir="verification/output/models"
def gen_error_for_evidence_effect(data_path,verification_result_top_dir):
    out_name="verification_error.csv"
    text_image_verification_dir=os.path.join(verification_result_top_dir,"text_image")
    text_verification_dir=os.path.join(verification_result_top_dir,"text")
    image_verification_dir=os.path.join(verification_result_top_dir,"image")
     
    save_verification(data_path,text_image_verification_dir,text_image_verification_dir,out_name)
    extract_error( data_path,text_image_verification_dir,out_name,None )
    save_verification(data_path,text_verification_dir,text_verification_dir,out_name)
    extract_error( data_path,text_verification_dir,out_name,None )
    save_verification(data_path,image_verification_dir,image_verification_dir,out_name)
    extract_error( data_path,image_verification_dir,out_name,None )
    
def gen_error_claim_id_in_text_image(text_image_verification_dir,error_name):
    corpus= os.path.join(text_image_verification_dir,error_name)
 
    corpus_df = pd.read_csv(corpus ,encoding="utf8")     
    total_claim_id_list=list(set(corpus_df['claim_id'].tolist()))
    return total_claim_id_list 
    
def check_dif(text_verification_dir,error_claim_id_in_text_image,out_name,error_name,sample_num):    
    corpus= os.path.join(text_verification_dir,error_name)
    corpus_df = pd.read_csv(corpus ,encoding="utf8")  
    print(f"before dif: {len(set(corpus_df['claim_id'].tolist()))}")
    corpus_df=corpus_df[~corpus_df['claim_id'].isin(error_claim_id_in_text_image)]
    corpus_df =sample_corpus(corpus_df,sample_num)
    corpus_df.to_csv(os.path.join(text_verification_dir,out_name),index=False)
    print(f"after dif: {len(set(corpus_df['claim_id'].tolist()))}")
    
def check_evidence_effect_analysis(data_path,verification_result_top_dir):
    gen_error_for_evidence_effect(data_path,verification_result_top_dir)
    out_name="verification_evidence_effect.csv"
    error_name="verification_error.csv"
    text_image_verification_dir=os.path.join(verification_result_top_dir,"text_image")
    text_verification_dir=os.path.join(verification_result_top_dir,"text")
    image_verification_dir=os.path.join(verification_result_top_dir,"image")
    error_claim_id_in_text_image=gen_error_claim_id_in_text_image(text_image_verification_dir,error_name)
    
    check_dif(text_verification_dir,error_claim_id_in_text_image,out_name,error_name,300)
    print("image")
    check_dif(image_verification_dir,error_claim_id_in_text_image,out_name,error_name,300)
     
     
     
    