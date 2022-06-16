

#/data/outputs/{}_{}_{}.{}.txt   
 
from datasets import   load_metric
import numpy as np 
import nltk 
from statistics import mean
import torch 
import json 

import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import BartTokenizer
import pandas as pd 
max_length_for_our_server=900



def get_data_with_evidence(data_file):
  
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
        if 'Evidence' in row.keys():
            evidence=row['Evidence']
        else:
            evidence=""
        
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
        evidences.append(evidence)
    return cleaned_ruling_outline_data, truth_claim_evidence_data , labels,evidences,[refuted,supported,nei]
               
def calc_metrics_without_generation( reference_csv_file,metric_file,max_target_length ):
    #get data
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    cleaned_ruling_outline_data, truth_claim_evidence_data  , labels,evidences,label_statistic = get_data_with_evidence(reference_csv_file )
    total_reference_token_id_list=[tokenizer.encode(cleaned_ruling_outline,  
                                               add_special_tokens=True   # Adds [CLS] and [SEP] token to every input text
                                                ,
                                                max_length= max_target_length, 
                                                truncation_strategy="longest_first",
                                               truncation =True
                                               )
                            for cleaned_ruling_outline in cleaned_ruling_outline_data]
    total_reference_text_list = [tokenizer.decode(g, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                        for g in total_reference_token_id_list]

    total_prediction_token_id_list=[tokenizer.encode(evidence,  
                                                add_special_tokens=True,   # Adds [CLS] and [SEP] token to every input text
                                                max_length= max_target_length, 
                                                truncation_strategy="longest_first",
                                               truncation =True)
                            for evidence  in evidences]  
    total_prediction_text_list = [tokenizer.decode(g, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                        for g in total_prediction_token_id_list]
    calc_metrics(total_reference_text_list,total_reference_token_id_list,total_prediction_text_list,total_prediction_token_id_list,
                 tokenizer,metric_file )
            
import pandas as pd             
import os   
from pathlib import Path
def merge_multi_style_prediction(prediction_file):
    path_object=Path(prediction_file)
    merged_df= pd.DataFrame(columns = ['predicted_explanation' , 'explanation' , 'truthfulness'])
    for style in ["refuted","supported","NEI"]:
        style_prediction_file=os.path.join(path_object.parent,f"{path_object.stem}_{style}{path_object.suffix}")
        df=pd.read_csv(style_prediction_file)
        frames = [merged_df, df ]
        merged_df = pd.concat(frames)
    merged_df.to_csv(prediction_file,index=False)
          
          
def get_gpu_id(cur_gpu_idx,gpu_list):
    gpu_num=len(gpu_list)
    
    if cur_gpu_idx%gpu_num==0:
        gpu=gpu_list[0]
    elif cur_gpu_idx%gpu_num==1:
        gpu=gpu_list[1]
    elif cur_gpu_idx%gpu_num==2:
        gpu=gpu_list[2] 
    else:
        gpu=gpu_list[3] 
    cur_gpu_idx+=1
    return gpu,cur_gpu_idx
          
def calc_metrics_for_all_setting(run_dir,gpu_list=["0,1,2,3"],is_merge="n",max_len=900):
    
    prediction_file=os.path.join(run_dir,"without/predict_all.csv")
    metric_file=os.path.join(run_dir,"without/metric.txt") 
    reference_file='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/Corpus2_for_controllable_generation_200.csv'

    after_retrieval_prediction_file=os.path.join(run_dir,"after_retrieval/predict_all.csv")     
    after_retrieval_metric_file=os.path.join(run_dir,"after_retrieval/metric.txt")  
    after_retrieval_reference_file='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/generation/Corpus2_for_controllable_generation_after_retrieval.csv'

    after_verification_prediction_file=os.path.join(run_dir,"after_verification/predict_all.csv")      
    after_verification_metric_file=os.path.join(run_dir,"after_verification/metric.txt")   
    after_verification_reference_file='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/generation/Corpus2_for_controllable_generation_after_verification.csv'

    after_retrieval_verification_prediction_file=os.path.join(run_dir,"after_retrieval_verification/predict_all.csv") 
    after_retrieval_verification_metric_file=os.path.join(run_dir,"after_retrieval_verification/metric.txt")  
    after_retrieval_verification_reference_file='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/generation/Corpus2_for_controllable_generation_after_retrieval_verification.csv'
    
    without_generation_after_verification_metric_file=os.path.join(run_dir,f"after_verification/without_generation_metric_{max_len}.txt")  
    without_generation_after_retrieval_verification_metric_file=os.path.join(run_dir,f"after_retrieval_verification/without_generation_metric_{max_len}.txt")  
    
    if is_merge=="y":
        merge_multi_style_prediction(prediction_file)
        merge_multi_style_prediction(after_retrieval_prediction_file)
        merge_multi_style_prediction(after_verification_prediction_file)
        merge_multi_style_prediction(after_retrieval_verification_prediction_file)
    
    print("without generation with gold evidence")
    calc_metrics_without_generation( after_verification_reference_file
                                    ,without_generation_after_verification_metric_file,max_len )
    print("without generation with system evidence")
    calc_metrics_without_generation( after_retrieval_verification_reference_file
                                    ,without_generation_after_retrieval_verification_metric_file,max_len )
    mp.set_start_method('spawn')
    processes = []
    cur_gpu_idx=0
     
    
    # print("without")
    # processes,cur_gpu_idx=set_mp(calc_metrics_from_file,prediction_file,
    #     reference_file,metric_file,max_len,gpu_list,processes,cur_gpu_idx)
    
 
    # print("after verification")
    
    # processes,cur_gpu_idx=set_mp(calc_metrics_from_file,after_verification_prediction_file,
    #     after_verification_reference_file,after_verification_metric_file,max_len ,gpu_list,processes,cur_gpu_idx)
    # print("after retrieval")
   
    # processes,cur_gpu_idx=set_mp(calc_metrics_from_file,after_retrieval_prediction_file,
    #     after_retrieval_reference_file,after_retrieval_metric_file,max_len  ,gpu_list,processes,cur_gpu_idx)
    # print("after retrieval verification")
 
    # processes,cur_gpu_idx=set_mp(calc_metrics_from_file,after_retrieval_verification_prediction_file,
    #     after_retrieval_verification_reference_file,after_retrieval_verification_metric_file,max_len ,gpu_list,processes,cur_gpu_idx)
    # for p in processes:
    #     p.join()
    print(f"finish  ")
     
def set_mp(func_name,prediction_file,
        reference_file,metric_file,max_len,gpu_list,processes,cur_gpu_idx):
    gpu,cur_gpu_idx =get_gpu_id(cur_gpu_idx,gpu_list) 
    os.environ['CUDA_VISIBLE_DEVICES']=gpu 
    p = mp.Process(target=func_name, args=(  prediction_file,
        reference_file,metric_file,max_len,))
            # We first train the model across `num_processes` processes
    p.start()
    processes.append(p)    
    return processes,cur_gpu_idx
    
    
def calc_metrics(total_reference_text_list,total_reference_token_id_list,total_prediction_text_list,total_prediction_token_id_list
                 ,tokenizer,metric_file  ):
    #tokenize
    # reference_list=word_tokenize_for_list(cleaned_ruling_outline_data )
    # tokenized_prediction_list=word_tokenize_for_list(prediction_list )
    #
    
    with open(metric_file,'w') as file:
        total_prediction_token_id_array=np.array([np.array(x) for x in total_prediction_token_id_list])
        total_reference_token_id_array=np.array([np.array(x) for x in total_reference_token_id_list])
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in total_prediction_token_id_array]
        reference_lens = [np.count_nonzero(ref != tokenizer.pad_token_id) for ref in total_reference_token_id_array]
        # total_prediction_token_id_list=torch.tensor(total_prediction_token_id_list) #TODO
        # total_reference_token_id_list=torch.torch(total_reference_token_id_list)
        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in total_prediction_token_id_list]
        # reference_lens = [np.count_nonzero(ref != tokenizer.pad_token_id) for ref in total_reference_token_id_list]
        gen_len =  round(np.mean(prediction_lens),4)
        ref_len =  round(np.mean(reference_lens),4)
        s=f"gen_len:{gen_len}, ref_len:{ref_len}"
        file.write(s+'\n')
        print(s)
        for metric in ["rouge","bleu","bertscore"]:#
            scores=calc_metric(total_prediction_text_list,total_reference_text_list,metric )
            json.dump(scores, file, indent=2)
            
            
         
# def count_num(total_text_token_list,tokenizer):
#     num=0
#     for text_token_list in total_text_token_list:
#         for token in text_token_list:
#             if token != tokenizer.pad_token_id:
#                 num+=1
#     return num/len(total_text_token_list)
    

def calc_metric(preds, labels ,metric_name ):
    metric = load_metric( metric_name)
    print( metric_name)
    
    if isinstance(preds, tuple):
        preds = preds[0]

    result=compute_metrics_with_text(preds,labels ,metric_name,metric)
    
    
    print(result)
    return result    
    
def compute_metric_with_text_bertscore(decoded_preds,decoded_labels,  metric):
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels )

    all_result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    avg_result=mean(all_result["f1"])
    result={"bertscore":avg_result}
    
    # Extract a few results from ROUGE
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()} TODO
    result = {key: value* 100 for key, value in result.items()} 
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def compute_metric_with_text_bleu(decoded_preds,decoded_labels ,metric):
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels )
    decoded_preds_bleu  = [pred.split(' ') for pred in decoded_preds]
    decoded_labels_bleu = [[pred.split(' ')] for pred in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds_bleu, references=decoded_labels_bleu )
    bleu_value=result["bleu"]
    bleu_value=round(bleu_value* 100,4)
    result["bleu"]=bleu_value
    # Extract a few results from ROUGE
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()} TODO
    
    return result

def compute_metrics_with_text(decoded_preds,decoded_labels, metric_name,metric):
    if metric_name=="bleu":
        return compute_metric_with_text_bleu(decoded_preds,decoded_labels,metric )
    elif metric_name=="bertscore":
        return compute_metric_with_text_bertscore(decoded_preds,decoded_labels ,metric)
    else:
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels )

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()} 
        result = {k: round(v, 4) for k, v in result.items()}
        return result



from nltk.tokenize.treebank import TreebankWordDetokenizer    
def postprocess_text(preds, labels ):
    
    len_limited_preds=preds
        
    preds = [pred.strip() for pred in len_limited_preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in len_limited_preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

# def compute_metrics_by_input(raw_datasets):
#     preds=raw_datasets["test"]["truth_claim_evidence"]
#     labels=raw_datasets["test"]["cleaned_ruling_outline"]
    
#     cur_result=compute_metrics_with_text(preds,labels, data_args.is_compute_metrics_by_input, data_args.num_limit)
#     # cur_result = {k: round(v, 4) for k, v in cur_result.items()} TODO
#     print(cur_result)


    # if data_args.is_compute_metrics_by_input=="y":
    #     compute_metrics_by_input(raw_datasets)
    #     return            
if __name__ == "__main__":
    calc_metrics_without_generation( '/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/generation/Corpus2_for_controllable_generation_after_retrieval_verification.csv'
                                    ,"controllable/generation/output/bart/run_test/bart_log.txt",400 )
    calc_metrics_without_generation( '/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/generation/Corpus2_for_controllable_generation_after_verification.csv'
                                    ,"controllable/generation/output/bart/run_test/bart_log.txt",400 )
    # calc_metrics_without_generation( '/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/Corpus2_for_controllable_generation.csv'
    #                                 ,"controllable/generation/output/bart/run_test/bart_log.txt",400 )
    # calc_metrics_from_file("data/outputs/bart_mocheg_0.0.txt",'/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/Corpus2_for_controllable_generation.csv'
    #                        ,"data/outputs/bart_log.txt","n", 100)
    