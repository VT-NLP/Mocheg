import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
from nltk import tokenize
from retrieval.utils.config import config

class SemanticSearcher:
    def __init__(self,bi_encoder_checkpoint,cross_encoder_checkpoint,no_rerank=False ) :
        # We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
        self.bi_encoder = SentenceTransformer(bi_encoder_checkpoint)
        self.bi_encoder.max_seq_length = 512     #Truncate long passages to 256 tokens
        #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
        self.cross_encoder = CrossEncoder(cross_encoder_checkpoint)
        self.no_rerank=no_rerank
        print(f"self.no_rerank: {self.no_rerank}")

    def encode_corpus(self,document_sent_list):
        
        # We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
        self.corpus_embeddings = self.bi_encoder.encode(document_sent_list, convert_to_tensor=True, show_progress_bar=True)
        self.passages=document_sent_list
    
    def search_batch(self,query_list,top_k):
        pass

    def search(self,query,top_k):
        if self.no_rerank:
            bi_encoder_top_k=top_k
        else:
            bi_encoder_top_k=1000
        ##### Sematic Search #####
        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k=bi_encoder_top_k)
        hits = hits[0]  # Get the hits for the first query
        hits_before_cross_encoder = sorted(hits, key=lambda x: x['score'], reverse=True) 
        
        if self.no_rerank:
            return hits_before_cross_encoder[:top_k]
        else:
            ##### Re-Ranking #####
            # Now, score all retrieved passages with the cross_encoder
            cross_inp = [[query, self.passages[hit['corpus_id']]] for hit in hits]
            cross_scores = self.cross_encoder.predict(cross_inp)

            # Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]

            
            hits_after_cross_encoder = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
            

            if config.verbose==True:
                print("Input question:", query)
                # Output of top-5 hits from bi-encoder
                print("\n-------------------------\n")
                print("Top-3 Bi-Encoder Retrieval hits")
                # hits = sorted(hits, key=lambda x: x['score'], reverse=True)
                for hit in hits[0:3]:
                    print("\t{:.3f}\t{}".format(hit['score'], self.passages[hit['corpus_id']].replace("\n", " ")))

                # Output of top-5 hits from re-ranker
                print("\n-------------------------\n")
                print("Top-3 Cross-Encoder Re-ranker hits")
                for hit in hits_after_cross_encoder[0:3]:
                    print("\t{:.3f}\t{}".format(hit['cross-score'], self.passages[hit['corpus_id']].replace("\n", " ")))

            
       
            return hits_after_cross_encoder[:top_k]