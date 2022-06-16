import torch 
import torch.nn as nn
import torch.nn.functional as F

from verification.util.enums import ModelAttribute



class StanceDetectionLayer(torch.nn.Module):
    def __init__(self, embed_dim,max_image_num,model_type,model_type_enum):
        super(StanceDetectionLayer, self).__init__()
        self.stance_detector = nn.Linear(embed_dim*2, 3)
        
        self.max_image_num=max_image_num
        self.model_type=model_type
        self.model_type_enum=model_type_enum
        
        
    
    def forward(self,  claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask):
        stance_embed=torch.tensor([],device=claim_embed.get_device())
        if self.model_type_enum.use_text_evidence:
            text_stance_embed=self.init_stance_embed(claim_embed,txt_evidences_embed)
            stance_embed=torch.cat([stance_embed,text_stance_embed],0)
        
        if self.model_type_enum.use_image_evidence and has_image:
            image_stance_embed=self.init_stance_embed(claim_embed,img_evidences_embed)
            stance_embed=torch.cat([stance_embed,image_stance_embed],0)
            
             
        stance_embed=self.stance_detector(stance_embed)
        stance_embed=torch.mean(stance_embed,0,True)
        return stance_embed

    def init_stance_embed(self,claim_embed,evidence_embed):
        evidences_num=evidence_embed.shape[0]
        duplicated_claim_embed=claim_embed.repeat(evidences_num,1 )
        stance_embed=torch.cat((duplicated_claim_embed,evidence_embed),-1)
        return stance_embed

class StanceDetectionLayer3(torch.nn.Module):
    def __init__(self, embed_dim,max_image_num,model_type,model_type_enum):
        super(StanceDetectionLayer3, self).__init__()
        self.fc2 = nn.Linear(embed_dim*2, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.3)
        self.max_image_num=max_image_num
        self.model_type=model_type
        self.model_type_enum=model_type_enum
        
        
    
    def forward(self,  claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask):
        stance_embed=torch.tensor([],device=claim_embed.get_device())
        if self.model_type_enum.use_text_evidence:
            text_stance_embed=self.init_stance_embed(claim_embed,txt_evidences_embed)
            stance_embed=torch.cat([stance_embed,text_stance_embed],0)
        
        if self.model_type_enum.use_image_evidence and has_image:
            image_stance_embed=self.init_stance_embed(claim_embed,img_evidences_embed)
            stance_embed=torch.cat([stance_embed,image_stance_embed],0)
            
             
        output=self.dropout(F.relu(self.fc2(stance_embed)))
        output= self.fc3(output) 
        output=torch.mean(output,0,True)
        return output

    def init_stance_embed(self,claim_embed,evidence_embed):
        evidences_num=evidence_embed.shape[0]
        duplicated_claim_embed=claim_embed.repeat(evidences_num,1 )
        stance_embed=torch.cat((duplicated_claim_embed,evidence_embed),-1)
        return stance_embed
class StanceDetectionLayer2(torch.nn.Module):
    def __init__(self, embed_dim,max_image_num,model_type,model_type_enum):
        super(StanceDetectionLayer2, self).__init__()
        
        self.max_image_num=max_image_num
        self.model_type=model_type
        self.model_type_enum=model_type_enum
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 1)
        self.fc2 = nn.Linear(embed_dim, 64)
        self.fc3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self,  claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask):
        stance_embed=torch.tensor([],device=claim_embed.get_device())
        if self.model_type_enum.use_text_evidence:
            stance_embed=torch.cat([stance_embed,txt_evidences_embed],0)
        if self.model_type_enum.use_image_evidence and has_image:
            stance_embed=torch.cat([stance_embed,img_evidences_embed],0) 
        attn_output, attn_output_weights = self.multihead_attn(claim_embed.unsqueeze(0), stance_embed.unsqueeze(0), stance_embed.unsqueeze(0))
        output=self.dropout(F.relu(self.fc2(attn_output)))
        output= self.fc3(output) 
        output=torch.squeeze(output, 0)
        return output
 

# BertMultiwayMatch from https://github.com/wilburOne/cosmosqa/#repo-details
class BertMultiwayMatch(torch.nn.Module):
    def __init__(self, img_embed_dim,text_embed_dim,model_type_enum ):
        super(BertMultiwayMatch, self).__init__( )
        hidden_size=text_embed_dim
        # self.dropout = nn.Dropout( hidden_dropout_prob)
        # self.linear_trans = nn.Linear( hidden_size,  hidden_size)
        self.linear_fuse_p = nn.Linear( hidden_size*2,  hidden_size)
        # self.linear_fuse_q = nn.Linear( hidden_size*2,  hidden_size)
        # self.linear_fuse_a = nn.Linear( hidden_size * 2,  hidden_size)
        self.classifier = nn.Linear( hidden_size,3)
        self.img_multihead_attn = nn.MultiheadAttention(text_embed_dim, 1,kdim=img_embed_dim,  vdim=img_embed_dim, batch_first=True)
        self.txt_multihead_attn = nn.MultiheadAttention(text_embed_dim, 1,kdim=text_embed_dim,  vdim=text_embed_dim, batch_first=True)
        self.model_type_enum=model_type_enum
    
    # sub and multiply
    def fusing_mlp_and_classify(self,  mclaim_text, claim_embed_for_text_evidence ,claim_mask ):
        new_mp_q = torch.cat([mclaim_text - claim_embed_for_text_evidence, mclaim_text * claim_embed_for_text_evidence], 2)
        
        # use separate linear functions
        new_mp_ = F.leaky_relu(self.linear_fuse_p(new_mp_q)) 
        new_p_max=self.gen_max( new_mp_,claim_mask)
        new_p_max = self.classifier(new_p_max)
 
        return new_p_max

    def gen_mask_and_shaped_claim(self,claim_encoded, evidence_encoded, claim_mask, evidence_mask, is_text): 
        # claim_mask_in_text_evidence_shape=repeat_claim_in_evidence_shape_2d(claim_mask,evidence_mask)
        claim_embed_for_text_evidence=repeat_claim_in_evidence_shape(claim_encoded,evidence_encoded)
        attention_mask=self.gen_attention_mask( evidence_mask)
        if is_text:
            mc_e, attn_output_weights  = self.txt_multihead_attn(claim_embed_for_text_evidence, evidence_encoded, evidence_encoded, key_padding_mask =attention_mask )
        else:
            mc_e, attn_output_weights  = self.img_multihead_attn(claim_embed_for_text_evidence, evidence_encoded, evidence_encoded,key_padding_mask =attention_mask)
        return mc_e,claim_embed_for_text_evidence 
        
    def gen_attention_mask(self,evidence_bert_mask):
        attention_mask = evidence_bert_mask.to(dtype=torch.bool )
        return ~attention_mask

    def gen_stance(self,  claim_encoded, evidence_encoded, claim_mask,evidence_mask,is_text):
        mc_t,claim_embed_for_text_evidence=self.gen_mask_and_shaped_claim(claim_encoded, evidence_encoded, claim_mask, evidence_mask, is_text)
        # MLP fuse
        new_claim_max_text=self.fusing_mlp_and_classify(   mc_t,  claim_embed_for_text_evidence,claim_mask )
        return new_claim_max_text

    def gen_max(self,new_mp_,claim_mask):
        #somehow tricky. before this step, we use nan value to be the claim mask; But we need to set it to -10000 to make max() work
        attn_mask=claim_mask.to(torch.bool)
        attn_mask=~attn_mask
        attn_mask=attn_mask.unsqueeze(-1)
        new_mp_=new_mp_.masked_fill_(attn_mask, -10000)
        new_p_max, new_p_idx = torch.max(new_mp_, 1)  
        return new_p_max

    def forward(self,  claim_encoded, text_evidence_encoded, image_evidence_encoded, has_image,claim_mask,txt_evidence_mask,img_evidence_mask):
        stance_embed=torch.tensor([],device=claim_encoded.get_device())
        if self.model_type_enum.use_text_evidence:
            text_stance_embed=self.gen_stance (   claim_encoded, text_evidence_encoded, claim_mask,txt_evidence_mask,True)
            stance_embed=torch.cat([stance_embed,text_stance_embed],0)
        
        if self.model_type_enum.use_image_evidence and has_image:
            image_stance=self.gen_stance (   claim_encoded, image_evidence_encoded, claim_mask,img_evidence_mask,False)
            stance_embed=torch.cat([stance_embed,image_stance],0)
        #TODO try cat img and text evidence
        logits=torch.mean(stance_embed,0,True)
        return logits
        
        
        #TODO try to also fusing_mlp for evidence embeddings and use them
        
        
def repeat_claim_in_evidence_shape_2d(claim_encoded, evidence_encoded):
    evidences_num=evidence_encoded.shape[0]
    duplicated_claim_embed=claim_encoded.repeat(evidences_num, 1 ) 
    return duplicated_claim_embed

        
def repeat_claim_in_evidence_shape(claim_encoded, evidence_encoded):
    evidences_num=evidence_encoded.shape[0]
    duplicated_claim_embed=claim_encoded.repeat(evidences_num,1, 1 ) 
    return duplicated_claim_embed




 # try to first avg then classifier for 3*hidden_size -> 3
class BertMultiwayMatch2(BertMultiwayMatch):
    def __init__(self, img_embed_dim,text_embed_dim,model_type_enum ):
        super(BertMultiwayMatch2, self).__init__(img_embed_dim,text_embed_dim ,model_type_enum)
        self.hidden_size=text_embed_dim
         
        self.classifier = nn.Linear( self.hidden_size*2,3) 
        self.classifier_text = nn.Linear( self.hidden_size,3) 
        

    def fusing_mlp_and_classify(self,  mclaim_text, claim_embed_for_text_evidence  ,claim_mask):
        new_mp_q = torch.cat([mclaim_text - claim_embed_for_text_evidence, mclaim_text * claim_embed_for_text_evidence], 2)

        # use separate linear functions
        new_mp_ = F.leaky_relu(self.linear_fuse_p(new_mp_q)) 
        new_p_max=self.gen_max( new_mp_,claim_mask)
        new_p_max=torch.mean(new_p_max,0,True)
        
       
        return new_p_max


    def forward(self,  claim_encoded, text_evidence_encoded, image_evidence_encoded, has_image,claim_mask,txt_evidence_mask,img_evidence_mask):
        stance_embed=torch.tensor([],device=claim_encoded.get_device())
        if self.model_type_enum.use_text_evidence:
            text_stance_embed=self.gen_stance (   claim_encoded, text_evidence_encoded, claim_mask,txt_evidence_mask,True)
            stance_embed=torch.cat([stance_embed,text_stance_embed],-1)
        
        if self.model_type_enum.use_image_evidence and has_image:
            image_stance=self.gen_stance (   claim_encoded, image_evidence_encoded, claim_mask,img_evidence_mask,False)
            stance_embed=torch.cat([stance_embed,image_stance],-1)
        
        if stance_embed.shape[-1]==self.hidden_size:
            logits = self.classifier_text(stance_embed)
        else:
            logits = self.classifier(stance_embed)
        
        return logits
        


 #  CLS
class BertMultiwayMatch3(BertMultiwayMatch2):
    def __init__(self, img_embed_dim,text_embed_dim,model_type_enum ):
        super(BertMultiwayMatch3, self).__init__(img_embed_dim,text_embed_dim,model_type_enum )
   
    def fusing_mlp_and_classify(self,  mclaim_text, claim_embed_for_text_evidence  ,claim_mask):
        new_mp_q = torch.cat([mclaim_text - claim_embed_for_text_evidence, mclaim_text * claim_embed_for_text_evidence], 2)

        # use separate linear functions
        new_mp_ = F.leaky_relu(self.linear_fuse_p(new_mp_q)) 
        cls_token_embed=new_mp_[:,0,:]
        # new_p_max=self.gen_max( new_mp_,claim_mask)
        new_p_max=torch.mean(cls_token_embed,0,True)
        
       
        return new_p_max

 