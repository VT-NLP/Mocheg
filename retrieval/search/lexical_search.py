# We also compare the results to lexical search (keyword search). Here, we use 
# the BM25 algorithm which is implemented in the rank_bm25 package.

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
from retrieval.utils.config import config

# We lower case our text and remove stop-words from indexing







class LexicalSearcher:
    def __init__(self )  :
        pass

    def encode_corpus(self, passages):
        tokenized_corpus = []
        self.passages=passages
        for passage in tqdm(passages):
            tokenized_corpus.append(self.tokenize(passage))
        self.bm25 = BM25Okapi(tokenized_corpus) 

        
    def tokenize(self,text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)

            if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                tokenized_doc.append(token)
        return tokenized_doc

    def search(self,query,top_k):
        

        ##### BM25 search (lexical search) #####
        bm25_scores = self.bm25.get_scores(self.tokenize(query))
        top_hits = np.argpartition(bm25_scores, -top_k)[-top_k:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_hits]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
        
        if config.verbose==True:
            print("Input question:", query)
            print("Top-3 lexical search (BM25) hits")
            for hit in bm25_hits[0:3]:
                print("\t{:.3f}\t{}".format(hit['score'], self.passages[hit['corpus_id']].replace("\n", " ")))   
        return bm25_hits