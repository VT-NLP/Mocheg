from IPython.display import display, HTML
  
import pandas as pd
import numpy as np

class Saver:
    def __init__(self) -> None:
        dict = {
        'claim_id':[],
        'Claim':[ ],
        'Evidence':[ ],
        'cleaned_truthfulness':[ ]
        }
        self.df=pd.DataFrame(dict)

    def add_retrieved_text(self,claim ,semantic_results,relevant_document_text_paragraph_list,truthfulness,claim_id,ruling_outline,ruling_article):
        retrieved_text=""
        for hit in semantic_results:
            retrieved_text+=" "+relevant_document_text_paragraph_list[hit['corpus_id']]
        df2 = {'claim_id':int(claim_id),'Claim': claim, 'Evidence': retrieved_text, 'cleaned_truthfulness': truthfulness,
               "ruling_outline":ruling_outline,"Origin":ruling_article}
        self.df = self.df.append(df2, ignore_index = True)
        # display(self.df)

    def save(self,out_path):
        self.df.to_csv(out_path,index=False)


    def insert_and_save(self,out_path,column_name,column_value_list):
        self.df=pd.read_csv(out_path)  
        self.df.insert(2, column_name, column_value_list)
        # self.df[column_name]=column_value_list
        self.df.to_csv(out_path,index=False)