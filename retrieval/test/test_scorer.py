

from transformers import pipeline 
import torch 
import numpy as np
 
from retrieval.search.image_search import ImageSearcher
from retrieval.search.lexical_search import LexicalSearcher
from retrieval.search.semantic_search import SemanticSearcher

from utils.data_util import *
import os 
from PIL import Image
import cv2

from utils.metrics import ImageScorer, TextScorer 




def test_text_scorer3():
    scorer=TextScorer()
    text_evidence_list =["Based on outbreaks of coronaviruses caused by animal to human transmissions such as in Asia in 2003 that caused Severe Acute Respiratory Syndrome (SARS), and in Saudi Arabia in 2012 that caused Middle East Respiratory Syndrome (MERS), NIH and the National Institute of Allergy and Infectious Diseases (NIAID) have for many years supported grants to learn more about viruses lurking in bats and other mammals that have the potential to spill over to humans and cause widespread disease. However, neither NIH nor NIAID have ever approved any grant that would have supported “gain-of-function” research on coronaviruses that would have increased their transmissibility or lethality for humans. NIH strongly supports the need for further investigation by the World Health Organization (WHO) into the origins of the SARS-CoV-2 coronavirus.","In accordance with English law, 19 of the victims of the Salem Witch Trials were instead taken to the infamous Gallows Hill to die by hanging. The elderly Giles Corey, meanwhile, was pressed to death with heavy stones after he refused to enter an innocent or guilty plea. Still more accused sorcerers died in jail while awaiting trial.","The myth of burnings at the stake in Salem is most likely inspired by European witch trials, where execution by fire was a disturbingly common practice."]
    relevant_document_text_list=["Based on  outbreaks of coronaviruses caused by animal to human transmissions such as in Asia in 2003 that caused Severe Acute Respiratory Syndrome(link is external) (SARS), and in Saudi Arabia in 2012 that caused Middle East Respiratory Syndrome(link is external) (MERS), NIH and the National Institute of Allergy and Infectious Diseases (NIAID) have for many years supported grants to learn more about viruses lurking in bats and other mammals that have the potential to spill over to humans and cause widespread disease. However, neither NIH nor NIAID have ever approved any grant that would have supported “gain-of-function” research on coronaviruses that would have increased their transmissibility or lethality for humans. NIH strongly supports the need for further investigation by the World Health Organization (WHO) into the origins of the SARS-CoV-2 coronavirus. Working with a cross-regional coalition of 13 countries(link is external), we urge the WHO to begin the second phase of their study without delay.","In January 1692, a group of young girls in Salem Village, Massachusetts became consumed by disturbing “fits” accompanied by seizures, violent contortions and bloodcurdling screams. A doctor diagnosed the children as being victims of black magic, and over the next several months, allegations of witchcraft spread like a virus through the small Puritan settlement. Twenty people were eventually executed as witches, but contrary to popular belief, none of the condemned was burned at the stake. In accordance with English law, 19 of the victims of the Salem Witch Trials were instead taken to the infamous Gallows Hill to die by hanging. The elderly Giles Corey, meanwhile, was pressed to death with heavy stones after he refused to enter an innocent or guilty plea. Still more accused sorcerers died in jail while awaiting trial.","The myth of burnings at the stake in Salem is most likely inspired by European witch trials, where execution by fire was a disturbingly common practice. Medieval law codes such as the Holy Roman Empire’s “Constitutio Criminalis Carolina” stipulated that malevolent witchcraft should be punished by fire, and church leaders and local governments oversaw the burning of witches across parts of modern day Germany, Italy, Scotland, France and Scandinavia. Historians have since estimated that the witch-hunt hysteria that peaked between the 15th and 18th centuries saw some 50,000 people executed as witches in Europe. Many of these victims were hanged or beheaded first, but their bodies were typically incinerated afterwards to protect against postmortem sorcery. Other condemned witches were still alive when they faced the flames, and were left to endure an excruciating death by burning and inhalation of toxic fumes."]
    hits=[{'corpus_id': 0, 'score': 0.3046852946281433, 'cross-score': -7.659981}, {'corpus_id': 1, 'score': 0.38031044602394104, 'cross-score': -7.7555566}, {'corpus_id': 2, 'score': 0.30892041325569153, 'cross-score': -7.913192}]
    precision,recall=scorer.precision_recall_by_similarity(hits,relevant_document_text_list,text_evidence_list)
    print(f"{precision.item()} {recall.item()}")
    
def test_text_scorer2():
    scorer=TextScorer()
    text_evidence_list = ['The cat sits outside',
                'A man is playing guitar',
                'The new movie is awesome']

    relevant_document_text_list = ['The dog plays in the garden',
                'A woman watches TV',
                'The new movie is so great']
    hits=[{'corpus_id': 0, 'score': 0.3046852946281433, 'cross-score': -7.659981}, {'corpus_id': 1, 'score': 0.38031044602394104, 'cross-score': -7.7555566}, {'corpus_id': 2, 'score': 0.30892041325569153, 'cross-score': -7.913192}]
    precision,recall=scorer.precision_recall_by_similarity(hits,relevant_document_text_list,text_evidence_list)
    print(f"{precision.item()} {recall.item()}")

def test_image_scorer():
    scorer=ImageScorer()
    """
    0-1 same
    2-2 similar
    3-3 same
    """
    relevant_document_text_list = ['00281-01920-01-.jpg',
                '00281-01920-06-.jpg',
                '00283-01932-12-64005156e8764b368090ae9313b6dbe8_xl.jpg',
                "00296-01999-02-60ca5a2520bd1300181c6ae4.jpg"]
    text_evidence_list = ['00281-proof-11-last-years-posters.jpg',
                '00281-proof-13-YAF-911-poster.jpg',
                '00283-proof-01-plane-crash-gender-reveal-cancun.jpg',
                "00296-proof-01-GettyImages-1233139107.jpg"]

    
    hits=[{'corpus_id': 0, 'score': 0.3046852946281433, 'cross-score': -7.659981}, {'corpus_id': 1, 'score': 0.38031044602394104, 'cross-score': -7.7555566}, {'corpus_id': 2, 'score': 0.30892041325569153, 'cross-score': -7.913192},{'corpus_id': 3, 'score': 0.3046852946281433, 'cross-score': -7.659981}]
    image_folder="/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mode3_latest_v2/images"
    precision,recall=scorer.precision_recall_by_similarity(hits,relevant_document_text_list,text_evidence_list,image_folder)
    print(f"{precision.item()} {recall.item()}")

if __name__ == "__main__":
    
    test_image_scorer()  