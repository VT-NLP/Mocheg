from controllable.generation.util.analyse_len import calc
from controllable.generation.util.preprocess import merge_evidence

import os
import pandas as pd
from transformers import BartTokenizer
from itertools import zip_longest    
    
def preprocess_for_generation(data_path,in_name,out_name="Corpus2_for_generation.csv"):    
    preprocess_for_generation_one_subset( data_path+"/val",in_name,out_name)
    print("finish val")
    preprocess_for_generation_one_subset( data_path+ "/train",in_name,out_name)
    print("finish train")
    
    preprocess_for_generation_one_subset(  data_path+"/test",in_name,out_name)
    calc('facebook/bart-large',"",data_path,
     out_name)
    
def clean_to_four_columns(data_path,out_name): 
    
    corpus=os.path.join(data_path,out_name)
    out_corpus=os.path.join(data_path,out_name)
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    evidence_df=evidence_df[["truth_claim_evidence","cleaned_ruling_outline","cleaned_truthfulness","Evidence"]]
    evidence_df.to_csv(out_corpus,index=False)    
    

def clean_to_five_columns(data_path,out_name): 
    
    corpus=os.path.join(data_path,out_name)
    out_corpus=os.path.join(data_path,out_name)
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    evidence_df=evidence_df[["truth_claim_evidence","cleaned_ruling_outline","cleaned_truthfulness","Evidence"]]
    evidence_df.loc[evidence_df["cleaned_truthfulness"] =="refuted", "cleaned_truthfulness_int"] = 0
    evidence_df.loc[evidence_df["cleaned_truthfulness"] == "supported", "cleaned_truthfulness_int"] = 1
    evidence_df.loc[evidence_df["cleaned_truthfulness"] == "NEI", "cleaned_truthfulness_int"] = 2
    print(evidence_df.columns)
    evidence_df.to_csv(out_corpus,index=False)      
    
    
    
    
def preprocess_for_generation_one_subset(data_path,in_name,out_name):
    
    add_len_for_one_set( data_path ,in_name,out_name)
    merge_evidence_truthfulness( data_path,out_name,out_name)
    clean_to_five_columns( data_path,out_name)    


def drop_invalid_data_for_generation( evidence_df):
    
    return evidence_df

def add_len(data_path,in_name="Corpus2.csv",out_name="Corpus2_for_controllable_generation_merged_with_len.csv"):    
    add_len_for_one_set( data_path+"/val",out_name,out_name)
    print("finish val")
    add_len_for_one_set( data_path+ "/train",out_name,out_name)
    print("finish train")
    add_len_for_one_set(  data_path+"/test",out_name,out_name)
     

 
def group_elements(n, iterable, padvalue="pad"):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def remove_padvalue(batch_sent, padvalue="pad"):
    end_idx=len(batch_sent)
    for idx,sent in enumerate(batch_sent): 
        if sent ==padvalue:
            end_idx=idx
            break 
            
    return batch_sent[:end_idx]

def get_token_len_list(text_list_with_na,tokenizer):
    token_list=[]
    text_list =[]
    for text in text_list_with_na :
        if pd.isna(text):
            text_list.append("")
        else:
            text_list.append(text )
    padvalue="pad"            
    for batch_sent in group_elements(1024,text_list):
        if batch_sent[1023] ==padvalue:
            batch_sent=remove_padvalue(batch_sent)
        cur_token_list=tokenizer(batch_sent, 
                                add_special_tokens=False,   # Adds [CLS] and [SEP] token to every input text
                                                )
        token_list.extend(cur_token_list["input_ids"])
        
    token_len_list=[]
    for token in token_list:
        token_len_list.append(len(token))
    return token_len_list 

def add_len_for_one_set(data_path,in_name,out_name,need_merge=True):
    corpus_name=in_name
    if need_merge:
        merge_evidence( data_path,out_name,drop_invalid_data_for_generation)
        corpus_name=out_name
    corpus=os.path.join(data_path,corpus_name)
    out_corpus=os.path.join(data_path,out_name)
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    ruling_outline_token_len_list=get_token_len_list( evidence_df['ruling_outline'].tolist(),tokenizer)
    evidence_token_len_list=get_token_len_list( evidence_df['Evidence'].tolist(),tokenizer)
    ruling_article_token_len_list=get_token_len_list( evidence_df['Origin'].tolist(),tokenizer)
    
    evidence_df.insert(5, "evidence_token_len",evidence_token_len_list )
    evidence_df.insert(6, "ruling_article_token_len",ruling_article_token_len_list )
    evidence_df.insert(7, "ruling_outline_token_len",ruling_outline_token_len_list )
    evidence_df.to_csv(out_corpus,index=False)
    

def check_is_delete_batch(evidence_df  ):
    evidence_df=evidence_df.drop(evidence_df[(evidence_df['Origin'].isna()) & (evidence_df['ruling_outline'].isna())].index)  
    evidence_df=evidence_df.drop(evidence_df[(evidence_df['Origin'].isna()) & (evidence_df['evidence_token_len']<5)].index)  
    evidence_df=evidence_df.drop(evidence_df[(evidence_df['evidence_token_len']<5) & (evidence_df['ruling_outline'].isna())].index)
    
    # evidence_df=evidence_df.drop(evidence_df[(evidence_df['Evidence'].isna())  ].index)
    # evidence_df=evidence_df.drop(evidence_df[(evidence_df['ruling_outline'].isna())  ].index)
    # evidence_df=evidence_df.drop(evidence_df[(evidence_df['evidence_token_len']<5) & (evidence_df['ruling_article_token_len']>1024)].index)
    evidence_df=evidence_df.drop(evidence_df[(evidence_df['ruling_outline'].isna()) & (evidence_df['ruling_article_token_len']>800)].index)
    return evidence_df 
    
    
def check_is_delete(evidence,ruling_article,ruling_outline,tokenizer):
     
    # token=tokenizer(ruling_article,  
    #                                     add_special_tokens=True    # Adds [CLS] and [SEP] token to every input text
    #                                 #     ,
    #                                 #     max_length= max_length_for_model , 
    #                                 #     truncation_strategy="longest_first",
    #                                 #    truncation =True
    #                                     )
    # token_len=len(token_list)
    # # if pd.isna(ruling_outline) and token_len>1024:
    # #     return True 
    # if pd.isna(evidence) and token_len>1024:
    #     return "y" 
    
    return "n" 
    
    
def get_substitute(ruling_article,max_len,tokenizer,ruling_article_token_len):
    if ruling_article_token_len>max_len:
        token=tokenizer(ruling_article,  
                                        add_special_tokens=True    # Adds [CLS] and [SEP] token to every input text
                                    #     ,
                                    #     max_length= max_length_for_model , 
                                    #     truncation_strategy="longest_first",
                                    #    truncation =True
                                        )["input_ids"]
        new_token=token[-max_len:]
        substitute=tokenizer.decode(new_token, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    else:
        substitute=ruling_article
    return substitute
    
        
        
    
def merge_evidence_truthfulness(data_path,out_file_name,in_file_name):
    corpus=os.path.join(data_path,in_file_name)
    out_corpus=os.path.join(data_path,out_file_name)
    evidence_list=[]
    ruling_outline_list=[]
    is_delete_list=[]
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    evidence_df=check_is_delete_batch(evidence_df )
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    max_len=200
    for i,row in evidence_df.iterrows():
        truthfulness=row["cleaned_truthfulness"]
        claim=row["Claim"]
        evidence=row["Evidence"]
        ruling_article=row["Origin"]
        ruling_outline=row["ruling_outline"]
        ruling_article_len=row['ruling_article_token_len']
        
        is_delete=check_is_delete(evidence,ruling_article,ruling_outline,tokenizer)
        is_delete_list.append(is_delete)
        if pd.isna(ruling_outline) or len(ruling_outline)<=0: #for generation
            ruling_outline=get_substitute(ruling_article,max_len,tokenizer,ruling_article_len)
        elif row['evidence_token_len']<5  :
            evidence=get_substitute(ruling_article,max_len,tokenizer,ruling_article_len)
        ruling_outline_list.append(ruling_outline)
        truth_claim_evidence=truthfulness+" </s> "+claim+" </s> "+evidence
        evidence_list.append(truth_claim_evidence)
        # print(evidence)
    #cleaned_evidence column
    evidence_df.insert(1, "truth_claim_evidence",evidence_list )
    evidence_df.insert(2, "cleaned_ruling_outline",ruling_outline_list )
    evidence_df.insert(3, "is_delete",is_delete_list )
    evidence_df=evidence_df[evidence_df["is_delete"]=="n"]
    evidence_df.to_csv(out_corpus,index=False)


def clean_to_three_columns(data_path,out_name): 
 
    corpus=os.path.join(data_path,out_name)
    out_corpus=os.path.join(data_path,out_name)
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    evidence_df=evidence_df[["truth_claim_evidence","cleaned_ruling_outline","cleaned_truthfulness"]]
    evidence_df.to_csv(out_corpus,index=False)

def clean_to_two_columns(data_path,out_name): 
 
    corpus=os.path.join(data_path,out_name)
    out_corpus=os.path.join(data_path,out_name)
    evidence_df = pd.read_csv(corpus ,encoding="utf8")  
    evidence_df=evidence_df[["truth_claim_evidence","cleaned_ruling_outline"]]
    evidence_df.to_csv(out_corpus,index=False)
    
# def preprocess_for_controllable_generation(data_path):
#     in_name= "Corpus2_for_generation.csv"
#     out_name="Corpus2_for_controllageneration.csv"
#     dataset_name="em"
    
    
    
    
    
        
import argparse
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help=" ",default="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2")
    args = parser.parse_args()
    return args
 

if __name__ == '__main__':
    args = parser_args()
    preprocess_for_generation(args.data_path,"Corpus2_for_controllable_generation.csv")
    