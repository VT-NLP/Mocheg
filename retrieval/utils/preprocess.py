import os 
import pandas as pd 
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from transformers import CLIPTokenizer
def generate_corpus3_for_retrieval(data_path):
    ORIGIN_LINK_CORPUS="Corpus3.csv"
    df_evidence = pd.read_csv(os.path.join(data_path,ORIGIN_LINK_CORPUS) ,encoding="utf8")
    claim_id_list=[]
    relevant_document_id_list=[]
    paragraph_id_list=[]
    paragraph_list=[]
    for _,row in df_evidence.iterrows():
        claim_id=row["claim_id"]
        relevant_document_id=row["relevant_document_id"]
        relevant_document=row["Origin Document"]
        relevant_document=relevant_document.replace("<p>","")
        relevant_document=relevant_document.replace("</p>","")
        if not pd.isna(relevant_document):
            cur_paragraph_list=nltk.tokenize.sent_tokenize(relevant_document)
            paragraph_id=0
            for paragraph in cur_paragraph_list:
                # if relevant_document_id==121125 and paragraph_id==85:
                #     print("here")
                if not pd.isna(paragraph) and paragraph !="N/A":
                    claim_id_list.append(claim_id)
                    relevant_document_id_list.append(relevant_document_id)
                    paragraph_list.append(paragraph)
                    paragraph_id_list.append(paragraph_id)
                    paragraph_id+=1
            
    df = pd.DataFrame({"claim_id":claim_id_list,"relevant_document_id":relevant_document_id_list,
                       "paragraph_id":paragraph_id_list,"paragraph":paragraph_list})
    df.to_csv(os.path.join(data_path, "supplementary","Corpus3_sentence_level.csv"),index=False)

def generate_corpus_id_corpus3_for_retrieval(data_path):    
    df_evidence = pd.read_csv(os.path.join(data_path, "supplementary","Corpus3_sentence_level.csv") ,encoding="utf8")
    corpus_id_list=[]
    for _,row in df_evidence.iterrows():
        claim_id=row["claim_id"]
        relevant_document_id=row["relevant_document_id"]
        paragraph_id=row["paragraph_id"]
        corpus_id=str(claim_id)+"-"+str(row["relevant_document_id"])+"-"+str(row["paragraph_id"])
        corpus_id_list.append(corpus_id)
    df_evidence.insert(3, "corpus_id",corpus_id_list )
    df_evidence.to_csv(os.path.join(data_path, "supplementary","Corpus3_sentence_level.csv"),index=False)
    
class Paragraph:
    def __init__(self, corpus_id,paragraph_text):
        paragraph_text_list = word_tokenize(paragraph_text)  
        sw = stopwords.words('english')  
        self.token_set = {w for w in paragraph_text_list if not w in sw}  
        self.corpus_id=corpus_id 
        self.paragraph_text=paragraph_text 
        self.token_set_len=len(self.token_set)
        
        
         
        
def load_corpus3_for_retrieval(data_path):
    corpus_file=os.path.join(data_path, "supplementary","Corpus3_sentence_level.csv")
    df_corpus = pd.read_csv(corpus_file,encoding="utf8")
    corpus_dic={}
    for _,row in df_corpus.iterrows():
        claim_id=row["claim_id"]
        corpus_id=str(claim_id)+"-"+str(row["relevant_document_id"])+"-"+str(row["paragraph_id"])
        paragraph=row["paragraph"]
        if claim_id not in corpus_dic:
            relevant_dic={}
            relevant_dic[corpus_id]= Paragraph(corpus_id,paragraph)
            corpus_dic[claim_id]=relevant_dic    
        else:
            relevant_dic=corpus_dic[claim_id]
            relevant_dic[corpus_id]=Paragraph(corpus_id,paragraph)    
    return corpus_dic    
    
def generate_rel(data_path):
    corpus_dic=load_corpus3_for_retrieval(data_path)
    
    rel_docs_list=[] 
    query_file=os.path.join(data_path,"Corpus2.csv")
    df_evidence = pd.read_csv(query_file ,encoding="utf8")
    cur_claim_id=0
    negative_corpus_id_list=record_negative_corpus_id( relevant_dic)
    
    for idx,row in df_evidence.iterrows():
        claim_id=row["claim_id"]
        evidence=row["Evidence"]
        if idx%100==0:
            print(f"{idx}/{len(df_evidence)}")
            save(rel_docs_list,data_path)
        if claim_id in corpus_dic.keys():
            if not pd.isna(evidence):
                evidence=evidence.replace("<p>","")
                evidence=evidence.replace("</p>","")
                relevant_dic=corpus_dic[claim_id]
                evidence_sent_list=nltk.tokenize.sent_tokenize(evidence)
                has_found=False
                for corpus_id,paragraph in relevant_dic.items():
                    if isin(paragraph, evidence_sent_list):
                        has_found=True
                        rel_docs_list.append([claim_id,0,corpus_id,1])
                if has_found:
                    if claim_id!=cur_claim_id:
                        rel_docs_list=add_negative_corpus_id(negative_corpus_id_list,rel_docs_list,claim_id)
                        negative_corpus_id_list=record_negative_corpus_id( relevant_dic)
                        cur_claim_id=claim_id
            
    save(rel_docs_list,data_path)            
    

def save(rel_docs_list,data_path):
    df = pd.DataFrame(rel_docs_list, columns = ['TOPIC', 'ITERATION','DOCUMENT#','RELEVANCY'])
    df.to_csv(os.path.join(data_path,"text_evidence_qrels_sentence_level.csv"),index=False)#qrels.csv
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
                
def record_negative_corpus_id( relevant_dic):
    negative_corpus_id_list=[]
    length=100 if len(relevant_dic.keys())>100 else len(relevant_dic.keys())
    negative_corpus_id_list.extend(list(relevant_dic.keys())[:length])
     
    return negative_corpus_id_list
                     
def add_negative_corpus_id(negative_corpus_id_list,rel_docs_list,claim_id):
    for negative_corpus_id in negative_corpus_id_list:
        rel_docs_list.append([claim_id,0,negative_corpus_id,0])
    return rel_docs_list
                    
def isin(paragraph, evidence_sent_list):
    for evidence_sent in evidence_sent_list:
        if not pd.isna(evidence_sent):
            if similarity(evidence_sent,paragraph.token_set,paragraph.token_set_len)>0.85:
                return True 
    return False
##https://www.codespeedy.com/find-sentence-similarity-in-python/
def similarity(X,Y_token_set,Y_token_set_len):
   
    
    
    
    # X = X.lower() 
    # Y = Y.lower() 
    
    X_list = word_tokenize(X)  
   
    
    sw = stopwords.words('english')  

    
    X_set = {w for w in X_list if not w in sw}  
    
        
        
    inter_set=X_set & Y_token_set
    # rvector = X_set.union(Y_set)  
    # for w in rvector: 
    #     if w in X_set: l1.append(1)
    #     else: l1.append(0) 
    #     if w in Y_set: l2.append(1) 
    #     else: l2.append(0) 
    # c = 0
        
    # for i in range(len(rvector)): 
    #         c+= l1[i]*l2[i] 
    c=len(inter_set)
    cosine = c / float((len(X_set)+Y_token_set_len)*0.5) 
    # print("similarity: ", cosine)
    return cosine
        
def preprocess_truncate_claim(data_path):
    query_file=os.path.join(data_path,"Corpus2.csv")
    df_evidence = pd.read_csv(query_file ,encoding="utf8")
    tokenizer_for_truncation=  CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")     
    for idx,row in df_evidence.iterrows():
        claim=row["Claim"]
        if idx%100==0:
            print(f"{idx}/{len(df_evidence)}")
        query= truncate_text(claim,tokenizer_for_truncation)   
        df_evidence.loc[idx,'Claim'] = query
        # row["Claim"]=query
        
    df_evidence.to_csv(os.path.join(data_path,"Corpus2_for_retrieval.csv"),index=False)
            
            

def truncate_text( text,tokenizer_for_truncation):
    tokens= tokenizer_for_truncation([text],truncation =True ) # ,truncation =True  truncation to model max len
    # if len(tokens.input_ids[0])>77:
    #     print("here")
    decoded_text= tokenizer_for_truncation.decode(tokens.input_ids[0],skip_special_tokens =True) #[:77] 
    return decoded_text 
                
# print("test")     
# generate_rel("/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test")   
    
# print("train")
# generate_rel("/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/train")   

# print("val")
# generate_rel("/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/val")   

# similarity("During the argument, one of the suspects got out of the car, assaulted the employee at the window, and then got back in the vehicle."
        #    ,"During the argument, one of the  got out of the car, assaulted the employee at the window and then got back in the vehicle.")
# generate_corpus_id_corpus3_for_retrieval("/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test")    
# generate_corpus_id_corpus3_for_retrieval("/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/val")    
# generate_corpus_id_corpus3_for_retrieval("/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/train")    
preprocess_truncate_claim("data/train")  
preprocess_truncate_claim("data/test")  
preprocess_truncate_claim("data/val")  