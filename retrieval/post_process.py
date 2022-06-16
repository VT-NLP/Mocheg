import os 
import pandas as pd 
def fix_name(data_path):
    corpus=os.path.join(data_path,"retrieval_result.csv")
    df = pd.read_csv(corpus ,encoding="utf8")  
    df=df.rename(columns={"claim": "Claim", "truthfulness": "cleaned_truthfulness"})
    df.to_csv(corpus,index=False)
   
   
def fix_truthfulness(data_path,corpus2_path):
    corpus=os.path.join(data_path,"retrieval_result.csv")
    df = pd.read_csv(corpus ,encoding="utf8")  
    df_corpus2 = pd.read_csv(os.path.join(corpus2_path,"Corpus2.csv") ,encoding="utf8")  
    cleaned_truthfulness_list=[]
    for i,row in df.iterrows():
        claim_id=row["claim_id"]
        found_df=df_corpus2[df_corpus2["claim_id"]==claim_id]
        if len(found_df)>0:
            cleaned_truthfulness=found_df.head(1)["cleaned_truthfulness"].values[0]  
            cleaned_truthfulness_list.append(cleaned_truthfulness)
             
        else:
            print(f"{claim_id} wrong")
            return 
    df=df.drop(columns=['cleaned_truthfulness' ])
    df.insert(6, "cleaned_truthfulness",cleaned_truthfulness_list )
    df.to_csv(corpus,index=False)
    
    
    
fix_truthfulness("retrieval/output/evidence/train",'/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg/train')
fix_truthfulness("retrieval/output/evidence/val",'/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg/val')
fix_truthfulness("retrieval/output/evidence/test",'/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg/test')
    
# fix_name("retrieval/output/evidence/train")
# fix_name("retrieval/output/evidence/val")
# fix_name("retrieval/output/evidence/test")