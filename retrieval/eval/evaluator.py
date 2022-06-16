from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation
from typing import List, Tuple, Dict, Set, Callable
import torch
from torch import Tensor
class MultiMediaInformationRetrievalEvaluator(evaluation.InformationRetrievalEvaluator):
    def __init__(self,
                 queries: Dict[str, str],  #qid => query
                 corpus: Dict[str, str],  #cid => doc
                 relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 corpus_chunk_size: int = 50000,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10],
                 precision_recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': util.cos_sim, 'dot_score': util.dot_score},       #Score function, higher=more similar
                 main_score_function: str = None,
                 corpus_embed=None,
                 save_query_result_path=None
                 ):
        super().__init__(queries,corpus,
                 relevant_docs,
                 corpus_chunk_size,
                 mrr_at_k,
                 ndcg_at_k,
                 accuracy_at_k,
                 precision_recall_at_k,
                 map_at_k,
                 show_progress_bar,
                 batch_size,
                 name,
                 write_csv,
                 score_functions,
                 main_score_function 
                 )
        self.corpus_embed=corpus_embed
        self.save_query_result_path=save_query_result_path
        
    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None):
        if corpus_embeddings==None:
            corpus_embeddings=self.corpus_embed
        return super().compute_metrices(model,corpus_model,corpus_embeddings)
    
    
    def compute_metrics(self, queries_result_list: List[object]):
        if self.save_query_result_path !=None:
            save(queries_result_list,self.save_query_result_path)
        return super().compute_metrics(queries_result_list)
    
    
def save(queries_result_list,save_query_result_path):
    torch.save( queries_result_list ,save_query_result_path)    