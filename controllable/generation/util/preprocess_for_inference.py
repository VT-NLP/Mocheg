    
"""
Index(['Unnamed: 0', 'Unnamed: 0.1', 'claim_id', 'img_evidences', 'Claim',
       'evidences', 'cleaned_truthfulness'],
      dtype='object')
Index(['claim_id', 'Snopes URL', 'Commoncrawl URL', 'Offset', 'Length',
       'Category', 'SubCategory', 'Headline', 'Description', 'Source', 'Claim',
       'cleaned_truthfulness', 'Truthfulness', 'evidences', 'img_evidences',
       'Origin', 'ruling_outline', 'fact_checkor_website'],
      dtype='object')
"""    
import os 
import pandas as pd 
from controllable.generation.util.generation_preprocess import * 

def rename(evidence_df):
    evidence_df=evidence_df.rename(columns={"evidences":"Evidence"})
    return evidence_df

# Evidence <-evidences  retrieval_result.csv
def preprocess_for_with_retrieval(data_path,out_name):
    corpus=os.path.join(data_path,"retrieval/retrieval_result.csv")
    out_corpus=data_path+"/generation/"+out_name
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    evidence_df=rename(evidence_df)
    evidence_df.to_csv(out_corpus,index=False)
    
def int_truthfulness_to_str(verification_corpus_df):
    verification_corpus_df.loc[verification_corpus_df["cleaned_truthfulness"] == 0, "cleaned_truthfulness"] = "refuted"
    verification_corpus_df.loc[verification_corpus_df["cleaned_truthfulness"] == 1, "cleaned_truthfulness"] = "supported"
    verification_corpus_df.loc[verification_corpus_df["cleaned_truthfulness"] == 2, "cleaned_truthfulness"] = "NEI"
    
    
    return verification_corpus_df

  
     
def fix_retrieval_result_after_update_corpus2(data_path,update_file_name):
    corpus=os.path.join(data_path,"Corpus2.csv")
    evidence_df = pd.read_csv(corpus ,encoding="utf8")
    
    retrieval_result_corpus=os.path.join(data_path,update_file_name)
    retrieval_result_df = pd.read_csv(retrieval_result_corpus ,encoding="utf8")
    
    ruling_outline_list=[]
    ruling_article_list=[]
    retrieval_result_df =retrieval_result_df.drop(columns=[ 'Origin'])
    retrieval_result_df =retrieval_result_df.drop(columns=[ 'ruling_outline'])
    for i,row in retrieval_result_df.iterrows():
        claim_id=row["claim_id"]
        found=evidence_df.loc[evidence_df['claim_id'] ==claim_id]
        ruling_outline=found["ruling_outline"].values[0]
        ruling_article=found["Origin"].values[0]
        ruling_article_list.append(ruling_article)
        ruling_outline_list.append(ruling_outline)
  
    retrieval_result_df.insert(4, "ruling_outline",ruling_outline_list )
    retrieval_result_df.insert(5, "Origin",ruling_article_list )
    retrieval_result_df.to_csv(retrieval_result_corpus,index=False)     
     
#  Claim evidences Origin ruling_outline   retrieval_result.csv, cleaned_truthfulness, claim_id <- verification_result_after_retrieval.csv,
def preprocess_for_with_retrieval_verification(data_path,out_name):
    retrieval_corpus=os.path.join(data_path,"retrieval/retrieval_result.csv")
    verification_corpus=os.path.join(data_path,"verification/verification_result_after_retrieval.csv")
    out_corpus=data_path+"/generation/"+out_name
    
    retrieval_corpus_df = pd.read_csv(retrieval_corpus ,encoding="utf8")  
    verification_corpus_df = pd.read_csv(verification_corpus ,encoding="utf8")  
    verification_corpus_df=int_truthfulness_to_str(verification_corpus_df)
    retrieval_corpus_df=rename(retrieval_corpus_df)
    evidence_df=retrieval_corpus_df.merge(verification_corpus_df, how='inner',   on="claim_id",suffixes=(None,"_y")) 
    evidence_df=evidence_df.rename(columns={"cleaned_truthfulness":"cleaned_truthfulness_x"})
    evidence_df=evidence_df.rename(columns={"cleaned_truthfulness_y":"cleaned_truthfulness"})
    evidence_df.to_csv(out_corpus,index=False)
      
# cleaned_truthfulness, claim_id <-  verification_result.csv, Claim evidences Origin ruling_outline  Corpus2_for_verification
def preprocess_for_with_verification(data_path,out_name):
    retrieval_corpus=os.path.join(data_path,"Corpus2_for_verification.csv")
    verification_corpus=os.path.join(data_path,"verification/verification_result.csv")
    out_corpus=data_path+"/generation/"+out_name
    
    retrieval_corpus_df = pd.read_csv(retrieval_corpus ,encoding="utf8")  
    verification_corpus_df = pd.read_csv(verification_corpus ,encoding="utf8")  
    verification_corpus_df=int_truthfulness_to_str(verification_corpus_df)
    retrieval_corpus_df=rename(retrieval_corpus_df)
    evidence_df=retrieval_corpus_df.merge(verification_corpus_df, how='inner',   on="claim_id",suffixes=(None,"_y")) 
    evidence_df=evidence_df.rename(columns={"cleaned_truthfulness":"cleaned_truthfulness_x"})
    evidence_df=evidence_df.rename(columns={"cleaned_truthfulness_y":"cleaned_truthfulness"})
    evidence_df.to_csv(out_corpus,index=False) 
 



def preprocess_for_generation_inference(data_path):
    
    generation_data_path=data_path+"/generation"
    out_name="Corpus2_for_controllable_generation_after_retrieval.csv"
    preprocess_for_with_retrieval(data_path,out_name)
    add_len_for_one_set(  generation_data_path  ,out_name,out_name,False)
    merge_evidence_truthfulness( generation_data_path,out_name,out_name)
    clean_to_five_columns( generation_data_path,out_name)
    out_name="Corpus2_for_controllable_generation_after_retrieval_verification.csv"
    preprocess_for_with_retrieval_verification(data_path,out_name)
    add_len_for_one_set(  generation_data_path  ,out_name,out_name,False)
    merge_evidence_truthfulness( generation_data_path,out_name,out_name)
    clean_to_five_columns( generation_data_path,out_name)
    out_name="Corpus2_for_controllable_generation_after_verification.csv"
    preprocess_for_with_verification(data_path,out_name) 
    add_len_for_one_set(  generation_data_path  ,out_name,out_name,False)
    merge_evidence_truthfulness( generation_data_path,out_name,out_name)
    clean_to_five_columns( generation_data_path,out_name) 



# def 

# data_path="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg/test"
# preprocess(data_path)