import torch 
import torch.nn as nn
import torch.nn.functional as F

from verification.util.enums import ModelAttribute

class SentimentDetectionLayer2(torch.nn.Module):
    def __init__(self, input_dim ,token_pooling_method  ):
        super(SentimentDetectionLayer2, self).__init__()
       
        self.fc2 = nn.Linear(input_dim, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.3)
        self.token_pooling_method=token_pooling_method
    
    def forward(self,  claim_embed  ):
   
        output=self.dropout(F.relu(self.fc2(claim_embed)))
        output= self.fc3(output) 
        return output
    
    
class SentimentDetectionLayer5(SentimentDetectionLayer2):
    def __init__(self, input_dim  ,token_pooling_method ):
        super(SentimentDetectionLayer5, self).__init__(input_dim ,token_pooling_method )
       
     
    
    def forward(self,  claim_embed ):
    
        claim_embed, _ = torch.max(claim_embed, 1) 
        output=self.dropout(F.relu(self.fc2(claim_embed)))
        output= self.fc3(output) 
        
        return output
 

class SentimentDetectionLayer4(SentimentDetectionLayer2):
    def __init__(self, input_dim ,token_pooling_method  ):
        super(SentimentDetectionLayer4, self).__init__( input_dim ,token_pooling_method )
       
     
    
    def forward(self,  claim_embed  ):
  
        output=F.leaky_relu(self.fc2(claim_embed)) #self.dropout(F.relu(self.fc2(claim_embed)))
        output= self.fc3(output) 
        
        output=token_pooling(output,self.token_pooling_method)
         
        return output

def token_pooling(output,token_pooling_method):
    if  token_pooling_method=="max":
        output, _ = torch.max(output, 1) 
    elif  token_pooling_method=="avg":
        output  = torch.mean(output, 1) 
    else:
        output=output[:,0,:]
    return output 
    
class SentimentDetectionLayer(torch.nn.Module):
    def __init__(self, input_dim  ,token_pooling_method ):
        super(SentimentDetectionLayer, self).__init__()
        self.sentiment_detector =nn.Linear(input_dim, 3)
        
    
    def forward(self,  claim_embed  ):
        stance_embed=self.sentiment_detector(claim_embed)
        return stance_embed 
