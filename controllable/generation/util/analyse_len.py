

from transformers import BartTokenizer
import pandas as pd 
import os 
import numpy as np 
import matplotlib.pyplot as plt


def get_data(data_file,style=None):
  
    cleaned_ruling_outline_data = []
    truth_claim_evidence_data = []
    labels = []
    evidences=[]
    
    refuted=0
    supported=0
    nei=0
    df_data = pd.read_csv(data_file)
    for _,row in df_data.iterrows():
        true_label = row['cleaned_truthfulness']
        truth_claim_evidence = row['truth_claim_evidence']
        cleaned_ruling_outline = row['cleaned_ruling_outline']
        # evidence=row['Evidence']
        
        if style!=None and style!="all" and true_label!=style:
            continue 
        else:
            if true_label =="refuted":
                refuted+=1
                labels.append(0)
            elif true_label =="supported":    
                supported+=1
                labels.append(1)  
            elif true_label =="NEI":
                nei+=1
                labels.append(2)   
            else:
                print(f"error {true_label}")
                exit()        
        cleaned_ruling_outline_data.append(cleaned_ruling_outline.strip())
        truth_claim_evidence_data.append(truth_claim_evidence.strip())  
        # evidences.append(evidence)
    return cleaned_ruling_outline_data, truth_claim_evidence_data , labels,[refuted,supported,nei]
            
            

def calc_len_for_one_file(pre_trained_dir,prediction_txt_file,reference_csv_file,has_prediction):
    # print(f"calc_len_for_one_file:{os.path.basename(reference_csv_file)}")
    max_length_for_our_server=900
    max_length_for_model=1024
    tokenizer = BartTokenizer.from_pretrained(pre_trained_dir)
    if has_prediction:
        with open(prediction_txt_file, 'r') as f1_file :
            prediction_list = f1_file.readlines()
    
    # prediction_df=pd.read_csv(prediction_txt_file)
    # prediction_list=prediction_df["predicted_explanation"]
    # cleaned_ruling_outline_data=prediction_df["explanation"]
    cleaned_ruling_outline_data, truth_claim_evidence_data  , labels, label_statistic = get_data(reference_csv_file )
    total_reference_token_id_list=[tokenizer.encode(cleaned_ruling_outline,  
                                               add_special_tokens=True   # Adds [CLS] and [SEP] token to every input text
                                            #     ,
                                            #     max_length= max_length_for_our_server, 
                                            #     truncation_strategy="longest_first",
                                            #    truncation =True
                                               )
                            for cleaned_ruling_outline in cleaned_ruling_outline_data]
    total_evidence_token_id_list=[tokenizer.encode(evidence,  
                                                add_special_tokens=True    # Adds [CLS] and [SEP] token to every input text
                                            #     ,
                                            #     max_length= max_length_for_model , 
                                            #     truncation_strategy="longest_first",
                                            #    truncation =True
                                               )
                            for evidence  in truth_claim_evidence_data]
    if has_prediction:
        total_prediction_token_id_list=[tokenizer.encode(evidence,  
                                                add_special_tokens=True   # Adds [CLS] and [SEP] token to every input text
                                                # ,
                                                # max_length= max_len , 
                                            #     truncation_strategy="longest_first",
                                            #    truncation =True
                                               )
                            for evidence  in prediction_list]
        print("prediction")
        calc_len_statistic(total_prediction_token_id_list,tokenizer)
    print("evidence")
    calc_len_statistic(total_evidence_token_id_list,tokenizer)
    print("reference")
    calc_len_statistic(total_reference_token_id_list,tokenizer)
    
    
    
def calc_len_statistic(total_token_id_list,tokenizer):
    total_prediction_token_id_array=np.array([np.array(x) for x in total_token_id_list])
   
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in total_prediction_token_id_array]
    gen_avg_len =  round(np.mean(prediction_lens),4)
    gen_max_len=round(np.amax(prediction_lens),4)
    
    histogram,_=np.histogram(prediction_lens, bins=10)
    print(f"{gen_avg_len},{gen_max_len},{histogram}")
    
    
def calc(tokenizer_dir,prediction_dir,data_path,corpus_name ):
    corpus_name_after_retrieval="generation/Corpus2_for_controllable_generation_after_retrieval.csv"
    print("train")
    calc_num(os.path.join(data_path,"train"),corpus_name )
     
    print("val")
    calc_num(os.path.join(data_path,"val"),corpus_name )
     
    print("test")
    calc_num(os.path.join(data_path,"test"),corpus_name )
    
    
    print("train")
     
    calc_len(tokenizer_dir,prediction_dir,os.path.join(data_path,"train"),corpus_name,corpus_name_after_retrieval)
    print("val")
     
    calc_len(tokenizer_dir,prediction_dir,os.path.join(data_path,"val"),corpus_name,corpus_name_after_retrieval)
    print("test")
     
    calc_len(tokenizer_dir,prediction_dir,os.path.join(data_path,"test"),corpus_name,corpus_name_after_retrieval)
    
def calc_num(data_path,corpus_name ):
    df_evidence = pd.read_csv(os.path.join(data_path,corpus_name) ,encoding="utf8")
    print(f" claim: {len(df_evidence)} ")
        
def calc_len(pre_trained_dir,run_dir,data_path,corpus_name,corpus_name_after_retrieval):
    has_prediction=False   
    calc_len_with_one_max_len_setting(pre_trained_dir,run_dir,has_prediction,data_path,corpus_name,corpus_name_after_retrieval)
    
def calc_len_with_one_max_len_setting(pre_trained_dir,run_dir,has_prediction,data_path,corpus_name,corpus_name_after_retrieval):    
    # print(f"calc_len_with_one_max_len_setting: {run_dir}")
    
    prediction_file=os.path.join(run_dir,"without/predict.txt")
    after_retrieval_prediction_file=os.path.join(run_dir,"after_retrieval/predict.txt")     
    reference_file=os.path.join(data_path,corpus_name)
    after_retrieval_reference_file=os.path.join(data_path,corpus_name_after_retrieval)
 
    calc_len_for_one_file(pre_trained_dir,prediction_file,reference_file,has_prediction)
    # calc_len_for_one_file(pre_trained_dir,after_retrieval_prediction_file,after_retrieval_reference_file,has_prediction)
    # calc_len_for_one_file(pre_trained_dir,after_verification_prediction_file,after_verification_reference_file)
    # calc_len_for_one_file(pre_trained_dir,after_retrieval_verification_prediction_file,after_retrieval_verification_reference_file)
    
    

if __name__ == '__main__':
         
    calc('facebook/bart-large',"outputs/bart/00005-","/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2",
        "Corpus2_for_controllable_generation.csv")    
     