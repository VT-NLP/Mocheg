from enum import Enum,auto
import json

 
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)
    
class ModelAttribute(Enum):
    
    
    
    def __init__(self,use_text_evidence,use_image_evidence,stance_layer,sentiment_layer,is_project_embed,only_claim_setting,token_pooling_method):
        self.use_text_evidence=use_text_evidence
        self.use_image_evidence=use_image_evidence
        self.stance_layer=stance_layer
        self.sentiment_layer=sentiment_layer
        self.is_project_embed=is_project_embed
        self.only_claim_setting=only_claim_setting
        self.token_pooling_method=token_pooling_method