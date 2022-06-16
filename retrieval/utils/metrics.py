
from nltk import tokenize
import nltk 
import torch 
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from retrieval.utils.config import config
import os

class TextScorer:
    def __init__(self,bi_encoder_checkpoint,score_by_fine_tuned):
        if score_by_fine_tuned:
            self.model = SentenceTransformer(bi_encoder_checkpoint)#'all-MiniLM-L6-v2'
        else:
            self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    def gen_retrieved_document_list(self,hits,relevant_document_list):
        retrieved_document_list=[]
        for hit in hits:
            retrieved_document_list.append(relevant_document_list[hit['corpus_id']])
        return retrieved_document_list

    def precision_recall_by_similarity(self,hits,relevant_document_list,evidence_document_list): 
        
        retrieved_document_list=self.gen_retrieved_document_list(hits,relevant_document_list)
        #split paragraphs into sentences
        retrieved_document_sent_list,evidence_document_sent_list=self.split_into_sentence_list(retrieved_document_list,evidence_document_list)
        precision,recall=self.calc_score(retrieved_document_sent_list,evidence_document_sent_list)
        return precision,recall
    

    def calc_score(self,retrieved_document_list,evidence_document_list):
        
        #Compute embedding for both lists
        retrieved_embeddings = self.model.encode(retrieved_document_list, convert_to_tensor=True)
        evidence_embeddings = self.model.encode(evidence_document_list, convert_to_tensor=True)

        #Compute cosine-similarits
        cosine_scores = util.pytorch_cos_sim(retrieved_embeddings, evidence_embeddings)
        if config.verbose==True:
            print(cosine_scores)
        precision=torch.sum(torch.max(cosine_scores, 1).values)/len(retrieved_document_list)
        recall=torch.sum(torch.max(cosine_scores, 0).values)/len(evidence_document_list)

        return precision,recall

    def split_into_sentence_list(self,retrieved_document_list,evidence_document_list):
        retrieved_document_sent_list,document_evidence_sent_list=[],[]
        for retrieved_document in retrieved_document_list:
            retrieved_document_sent_list.extend(tokenize.sent_tokenize(retrieved_document))
        for document_evidence in evidence_document_list:
            document_evidence_sent_list.extend(tokenize.sent_tokenize(document_evidence))
        return retrieved_document_sent_list,document_evidence_sent_list



class ImageScorer(TextScorer):
    def __init__(self,image_encoder_checkpoint,score_by_fine_tuned)  :
        super().__init__(image_encoder_checkpoint,score_by_fine_tuned)
        if score_by_fine_tuned:
            self.model = SentenceTransformer(image_encoder_checkpoint)
        else:
            self.model = SentenceTransformer('clip-ViT-B-32')

    def precision_recall_by_similarity(self,hits,relevant_document_list,evidence_document_name_list,img_folder): 
        retrieved_document_name_list=self.gen_retrieved_document_list(hits,relevant_document_list)
        retrieved_document_list,evidence_document_list=get_images(retrieved_document_name_list,evidence_document_name_list,img_folder)
        precision,recall=self.calc_score(retrieved_document_list,evidence_document_list)
        return precision,recall



def get_images(retrieved_document_name_list,evidence_document_name_list,img_folder):
     
    retrieved_document_list=[Image.open(os.path.join(img_folder,filepath)) for filepath in retrieved_document_name_list]
    evidence_document_list=[Image.open(os.path.join(img_folder,filepath)) for filepath in evidence_document_name_list]
    return retrieved_document_list,evidence_document_list