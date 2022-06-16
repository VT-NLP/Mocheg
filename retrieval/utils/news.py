"""
#In theory, it should be:
    claim
    truthfulness
    relevant_doc_list
        img_list
        text
        evidence_list
            img_list
            text_list
    claim_id
    snope_url
    ruling_article
    """
import pandas as pd 

class News :
    """
    claim
    truthfulness
    relevant_doc_dict
        relevant_doc 
            img_list
            text
    evidence_dict
        img_list
        txt_list
    claim_id
    snope_url
    ruling_article
    """
    def __init__(self,claim_id,snopes_url,text_evidence,claim ,truthfulness,ruling_article,ruling_outline ) :
        self.claim=claim 
        self.truthfulness=truthfulness
        self.relevant_doc_dict={}
        self.evidence_dict={}
        self.evidence_dict["img_list"]=[]
        self.evidence_dict["txt_list"]=[]
        if not pd.isna(text_evidence) and len(text_evidence)>0:
            self.evidence_dict["txt_list"].append(text_evidence)
        self.claim_id=claim_id
        self.snopes_url=snopes_url
        self.ruling_article=ruling_article
        self.ruling_outline=ruling_outline
        


    def add_text_evidence(self,text_evidence):
        if not pd.isna(text_evidence) and len(text_evidence)>0:
            self.evidence_dict["txt_list"].append(text_evidence)

    def add_img_evidence(self,img_evidence):
        self.evidence_dict["img_list"].append(img_evidence)

    def add_relevant_doc(self,relevant_doc_text,relevant_doc_id):
        relevant_doc={}
        relevant_doc["text"]=relevant_doc_text
        relevant_doc["img_list"]=[]
        self.relevant_doc_dict[relevant_doc_id]=relevant_doc

    def add_relevant_doc_img(self,relevant_doc_img,relevant_doc_id):
        if relevant_doc_id in self.relevant_doc_dict:
            relevant_doc=self.relevant_doc_dict[relevant_doc_id]
            relevant_doc["img_list"].append(relevant_doc_img)
        else:
            relevant_doc={}
            relevant_doc["text"]=""
            relevant_doc["img_list"]=[relevant_doc_img]
            self.relevant_doc_dict[relevant_doc_id]=relevant_doc

    def get_text_evidence_list(self):
        return self.evidence_dict["txt_list"]

    def get_img_evidence_list(self):
        return self.evidence_dict["img_list"]