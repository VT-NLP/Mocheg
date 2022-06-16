from enum import Enum,auto
import json

 
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)
    
class ModelAttribute(Enum):
    CLAIM_TEXT_IMAGE = (True,True,"2","2","y","sentiment","max")
    CLAIM_IMAGE=(False,True,"2","2","y","sentiment","max")
    CLAIM_TEXT=(True,False,"2","2","y","sentiment","max")
    CLAIM=(False,False,"2","2","y","sentiment","max")
    CLAIM_TEXT_IMAGE_2FC = (True,True,"3","2","y","sentiment","max")
    CLAIM_IMAGE_2FC=(False,True,"3","2","y","sentiment","max")
    CLAIM_TEXT_2FC=(True,False,"3","2","y","sentiment","max")
    CLAIM_2FC=(False,False,"3","2","y","sentiment","max")
    CLAIM_TEXT_IMAGE_FC = (True,True,"1","1","y","sentiment","max")
    CLAIM_IMAGE_FC=(False,True,"1","1","y","sentiment","max")
    CLAIM_TEXT_FC=(True,False,"1","1","y","sentiment","max")
    CLAIM_FC=(False,False,"1","1","y","sentiment","max")
    CLAIM_TEXT_IMAGE_attention = (True,True,"4","4","n","sentiment","max")
    CLAIM_TEXT_attention=(True,False,"4","4","n","sentiment","max")
    CLAIM_IMAGE_attention=(False,True,"4","4","n","sentiment","max")
    CLAIM_attention=(False,False,"4","4","n","sentiment","max")
    CLAIM_attention_5=(False,False,"4","5","n","sentiment","max")
    CLAIM_TEXT_IMAGE_attention_5_4= (True,True,"5","4","n","sentiment","max")
    CLAIM_TEXT_attention_5_4=(True,False,"5","4","n","sentiment","max")
    CLAIM_IMAGE_attention_5_4=(False,True,"5","4","n","sentiment","max")
    CLAIM_TEXT_IMAGE_attention_6_4= (True,True,"6","4","n","sentiment","max")
    CLAIM_attention_6_stance=(False,False,"6","4","n","stance","max")
    CLAIM_attention_5_stance=(False,False,"5","4","n","stance","max")
    CLAIM_avg=(False,False,"4","4","n","sentiment","avg")
    
    
    def __init__(self,use_text_evidence,use_image_evidence,stance_layer,sentiment_layer,is_project_embed,only_claim_setting,token_pooling_method):
        self.use_text_evidence=use_text_evidence
        self.use_image_evidence=use_image_evidence
        self.stance_layer=stance_layer
        self.sentiment_layer=sentiment_layer
        self.is_project_embed=is_project_embed
        self.only_claim_setting=only_claim_setting
        self.token_pooling_method=token_pooling_method