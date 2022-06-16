from enum import Enum,auto
 
import json

 
class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)
    
class TrainAttribute(Enum):
    BI_ENCODER = ('multi-qa-MiniLM-L6-cos-v1', 100,20,256,"txt")
    IMAGE_MODEL =('clip-ViT-B-32',77 ,20,480,"img") 
    CROSS_ENCODER=( 'cross-encoder/ms-marco-MiniLM-L-6-v2',100,20,512,"txt")
    
    
    def __init__(self,model_name,max_seq_length,epoch,batch_size ,media):
        self.model_name=model_name
        self.max_seq_length=max_seq_length 
        self.epoch=epoch 
        self.batch_size=batch_size
        self.media=media