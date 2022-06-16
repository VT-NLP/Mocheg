"""
This file evaluates CrossEncoder on the TREC 2019 Deep Learning (DL) Track: https://arxiv.org/abs/2003.07820
TREC 2019 DL is based on the corpus of MS Marco. MS Marco provides a sparse annotation, i.e., usually only a single
passage is marked as relevant for a given query. Many other highly relevant passages are not annotated and hence are treated
as an error if a model ranks those high.
TREC DL instead annotated up to 200 passages per query for their relevance to a given query. It is better suited to estimate
the model performance for the task of reranking in Information Retrieval.
Run:
python eval_cross-encoder-trec-dl.py cross-encoder-model-name
"""
import gzip
from collections import defaultdict
import logging
import tqdm
import numpy as np
import sys
import pytrec_eval
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import os
import torch 
from retrieval.data.data_dealer import TextDataDealer
from retrieval.eval.eval_msmarco_mocheg import load_qrels

def load_data(data_folder,args):
    if  args.media=="txt":
        data_dealer=TextDataDealer()
    
    corpus_max_size=0
    corpus=data_dealer.load_corpus(args.test_data_folder,  corpus_max_size)
    dev_rel_docs,needed_pids,relevant_qid,_=load_qrels(data_folder, args.media)
    queries=data_dealer.load_queries(data_folder,relevant_qid)
    
    #Read which passages are relevant
    relevant_docs = defaultdict(lambda: defaultdict(int))
    for qid, dev_rel in dev_rel_docs.items():
        for pid in dev_rel:
            relevant_docs[str(qid)][str(pid)]=1
    
  

  
    return queries,relevant_docs,relevant_qid,corpus



def load_top_1000_corpus( top_candidate_corpus_path,queries,corpus):
    queries_result_list=torch.load(top_candidate_corpus_path) 
    
    passage_cand = {}
    query_id_list=list(queries.keys())
    for query_itr, score_dict_list in enumerate(queries_result_list ):
        top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
        top_hits=top_hits[:1000]
        qid=query_id_list[query_itr]
        for score_dict in top_hits:
            corpus_id=score_dict["corpus_id"]
            score=score_dict["score"]
            if qid not in passage_cand:
                passage_cand[qid] = []
            passage_cand[qid].append([corpus_id, corpus[corpus_id]])
    return passage_cand
# def load_top_k_corpus(data_folder):
#     # Read the top 1000 passages that are supposed to be re-ranked
#     passage_filepath = os.path.join(data_folder, 'msmarco-passagetest2019-top1000.tsv.gz')

#     if not os.path.exists(passage_filepath):
#         logging.info("Download "+os.path.basename(passage_filepath))
#         util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz', passage_filepath)

#     passage_cand = {}
#     with gzip.open(passage_filepath, 'rt', encoding='utf8') as fIn:
#         for line in fIn:
#             qid, pid, query, passage = line.strip().split("\t")
#             if qid not in passage_cand:
#                 passage_cand[qid] = []

#             passage_cand[qid].append([pid, passage])
#     return passage_cand        
    
def test_cross_encoder(args):
    model_name=args.model_name
    data_folder = args.test_data_folder
    # os.makedirs(data_folder, exist_ok=True)

    queries,relevant_docs,relevant_qid,corpus=load_data(data_folder,args)
    passage_cand=load_top_1000_corpus( args.top_candidate_corpus_path,queries,corpus)

    logging.info("Queries: {}".format(len(queries)))

    queries_result_list = []
    run = {}
    model = CrossEncoder(model_name, max_length=100)

    for qid in tqdm.tqdm(relevant_qid):
        query = queries[qid]

        cand = passage_cand[qid]
        pids = [c[0] for c in cand]
        corpus_sentences = [c[1] for c in cand]

        cross_inp = [[query, sent] for sent in corpus_sentences]

        if model.config.num_labels > 1: #Cross-Encoder that predict more than 1 score, we use the last and apply softmax
            cross_scores = model.predict(cross_inp, apply_softmax=True)[:, 1].tolist()
        else:
            cross_scores = model.predict(cross_inp).tolist()

        cross_scores_sparse = {}
        for idx, pid in enumerate(pids):
            cross_scores_sparse[pid] = cross_scores[idx]

        sparse_scores = cross_scores_sparse
        run[str(qid)] = {}
        for pid in sparse_scores:
            run[str(qid)][str(pid)] = float(sparse_scores[pid])

    # metrics=[]
    evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs,   pytrec_eval.supported_measures)#{'ndcg_cut.10'}
    results = evaluator.evaluate(run)

    print("Queries:", len(relevant_qid))
    # print(scores)
    # print("NDCG@10: {:.2f}".format(np.mean([ele["ndcg_cut_10"] for ele in results.values()])*100))
        

    # for query_id, query_measures in sorted(results.items()):
    #     for measure, value in sorted(query_measures.items()):
    #         print_line(measure, query_id, value)

    # Scope hack: use query_measures of last item in previous loop to
    # figure out all unique measure names.
    #
    # TODO(cvangysel): add member to RelevanceEvaluator
    #                  with a list of measure names.
    for measure in sorted(list(results.values())[0].keys()):
        print_line(
            measure,
            'all',
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure]
                 for query_measures in results.values()]))
    
    
def print_line(measure, scope, value):
    print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))    

if __name__ == "__main__":
    
    test_cross_encoder(None)     