
import os
import pandas as pd


def merge_evidence(data_path, out_name,drop_invalid_data_func):
    corpus=os.path.join(data_path,"Corpus2.csv")
    evidence_df = pd.read_csv(corpus ,encoding="utf8")
    evidence_df=drop_invalid_data_func(evidence_df)
    out_corpus=os.path.join(data_path,out_name)
    evidence_list=[]
    evidence_merged_df = evidence_df.drop_duplicates(subset='claim_id', keep="first")
    evidence_merged_df =evidence_merged_df.drop(columns=[ 'Evidence'])
    evidence_merged_df=evidence_merged_df.reset_index(drop=True)
    for i,row in evidence_merged_df.iterrows():
        claim_id=row["claim_id"]
        found=evidence_df.loc[evidence_df['claim_id'] ==claim_id]
        evidence=append_str(found["Evidence"].values)
        evidence=evidence.replace("<p>","")
        evidence=evidence.replace("</p>","")
        evidence_list.append(evidence)
        # print(evidence)
    evidence_merged_df.insert(13, "Evidence",evidence_list )
    evidence_merged_df.to_csv(out_corpus,index=False)
    
def append_str(evidence_array):
    evidence=""
    for i in range(len(evidence_array)):
        evidence+=str(evidence_array[i])+" "
    return evidence


    
 