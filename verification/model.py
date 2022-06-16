from verification.util.stance_detect import * 
from verification.util.sentiment import * 


# class MultiModal(torch.nn.Module):
    
#     def __init__(self, bert, clip, concat_hidden_size, final_hidden_size, num_classes,max_image_num,model_type,model_type_enum,is_project_embed ):
#         """
#         In the constructor we instantiate two nn.Linear modules and assign them as
#         member variables.
#         """
#         super(MultiModal, self).__init__()
        
#         self.clipmodel = clip
        
#         # self.bert = bert
        
#         self.fc1 = nn.Linear(8192, 512)
        
#         self.fc2 = nn.Linear(512, 64)
        
#         self.fc3 = nn.Linear(64, num_classes)
        
#         self.dropout = nn.Dropout(0.3)
#         if model_type_enum.stance_layer=="2":
#             self.stance_detect_layer=StanceDetectionLayer2(512,max_image_num,model_type,model_type_enum)
#         elif model_type_enum.stance_layer=="3":
#             self.stance_detect_layer=StanceDetectionLayer3(512,max_image_num,model_type,model_type_enum)
        # elif model_type_enum.stance_layer=="4": 
        #     self.stance_detect_layer= BertMultiwayMatch( 768,512)
        # elif model_type_enum.stance_layer=="5": 
        #     self.stance_detect_layer= BertMultiwayMatch2( 768,512)
        # elif model_type_enum.stance_layer=="6": 
        #     self.stance_detect_layer= BertMultiwayMatch3( 768,512)    
        # else:
        #     self.stance_detect_layer=StanceDetectionLayer(512,max_image_num,model_type,model_type_enum)
       
#         if model_type_enum.sentiment_layer =="4":
#             self.sentiment_detect_layer=SentimentDetectionLayer4(512)
#         elif model_type_enum.sentiment_layer =="2":
#             self.sentiment_detect_layer=SentimentDetectionLayer2(512)
#         elif model_type_enum.sentiment_layer =="5":
#             self.sentiment_detect_layer=SentimentDetectionLayer5(512)
#         else:
#             self.sentiment_detect_layer=SentimentDetectionLayer(512)
#         self.max_image_num=max_image_num
#         self.model_type=model_type
#         self.model_type_enum=model_type_enum
#         self.is_project_embed=is_project_embed
        


#     def forward(self,  claim_img_encod,device):
#         """
#         In the forward function we accept a Tensor of input data and we must return
#         a Tensor of output data. We can use Modules defined in the constructor as
#         well as arbitrary operators on Tensors.
#         """
         
#         if 'pixel_values' in claim_img_encod.keys():
#             has_image=True
#             img_encod = claim_img_encod['pixel_values'].squeeze(0).to(device)
#             claim_img_encod['pixel_values'] = img_encod
#         else:
#             has_image=False
#         claim_ids = claim_img_encod['input_ids'].squeeze(0).to(device)
#         claim_img_attention_masks = claim_img_encod['attention_mask'].squeeze(0).to(device)
#         claim_img_encod['input_ids'] = claim_ids
#         claim_img_encod['attention_mask'] = claim_img_attention_masks

#         if has_image:
#             output_clip = self.clipmodel(**claim_img_encod)
#             if self.is_project_embed =="y":
#                 text_embeds=output_clip.text_embeds
#                 img_evidences_embed=output_clip.image_embeds     #m, 512
#                 img_evidence_mask=None
                
#             else:  
#                 text_embeds=output_clip.text_model_output.last_hidden_state  #(batch_size, sequence_length, hidden_size) 12,28,512
#                 img_evidences_embed=output_clip.vision_model_output.last_hidden_state  #(batch_size, sequence_length, hidden_size) 1,50,768
#                 img_evidence_mask=torch.ones(img_evidences_embed.shape[0],img_evidences_embed.shape[1], dtype=img_evidences_embed.dtype, device=img_evidences_embed.device) 
    
#         else:
#             img_evidences_embed=None
#             img_evidence_mask=None
#             if self.is_project_embed =="y":
#                 text_embeds = self.clipmodel.get_text_features(**claim_img_encod)
#             else:   
#                 text_model_output=self.clipmodel.text_model(**claim_img_encod) 
#                 text_embeds=text_model_output.last_hidden_state 
#         claim_embed= text_embeds[0:1]        #1,sequence size, 512
#         claim_mask= claim_img_attention_masks[0:1]
#         txt_evidences_embed= text_embeds[1:]    #n,sequence size, 512
#         txt_evidence_mask=claim_img_attention_masks[1:]
        
#         if self.has_evidence(has_image):
#             output=self.stance_detect_layer(claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
#         else:
#             output=self.sentiment_detect_layer(claim_embed  )
#         return output
    
#     def has_evidence(self,has_image):
#         if not self.model_type_enum.use_text_evidence and (not  self.model_type_enum.use_image_evidence or not has_image):
#             return False
#         else:
#             return True

class MultiModal2(torch.nn.Module):
    
    def __init__(self, bert, clip, concat_hidden_size, final_hidden_size, num_classes,max_image_num,model_type,model_type_enum,is_project_embed ,only_claim_setting,token_pooling_method):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiModal2, self).__init__()
        
        self.clipmodel = clip
        self.max_image_num=max_image_num
        self.model_type=model_type
        self.model_type_enum=model_type_enum
        self.is_project_embed=is_project_embed
        self.verifier=Verifier( max_image_num,model_type,model_type_enum,only_claim_setting,token_pooling_method)
        

    def forward(self,  claim_img_encod,device):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
         
        if 'pixel_values' in claim_img_encod.keys():
            has_image=True
            img_encod = claim_img_encod['pixel_values'].squeeze(0).to(device)
            claim_img_encod['pixel_values'] = img_encod
        else:
            has_image=False
        claim_ids = claim_img_encod['input_ids'].squeeze(0).to(device)
        claim_img_attention_masks = claim_img_encod['attention_mask'].squeeze(0).to(device)
        claim_img_encod['input_ids'] = claim_ids
        claim_img_encod['attention_mask'] = claim_img_attention_masks



        if self.is_project_embed =="y":
            if has_image:
                output_clip = self.clipmodel(**claim_img_encod)
                text_embeds=output_clip.text_embeds
                img_evidences_embed=output_clip.image_embeds     #m, 512
                img_evidence_mask=None
            else:
                text_embeds = self.clipmodel.get_text_features(**claim_img_encod)
                img_evidences_embed=None
                img_evidence_mask=None
        else:
            if has_image:
                output_clip = self.clipmodel(**claim_img_encod)
                text_embeds=output_clip.text_model_output.last_hidden_state  #(batch_size, sequence_length, hidden_size) 12,28,512
                img_evidences_embed=output_clip.vision_model_output.last_hidden_state  #(batch_size, sequence_length, hidden_size) 1,50,768
                img_evidences_embed=img_evidences_embed[:,1:,:] #  remove CLS token    
                img_evidence_mask=torch.ones(img_evidences_embed.shape[0],img_evidences_embed.shape[1], dtype=img_evidences_embed.dtype, device=img_evidences_embed.device) 
            else:
                img_evidences_embed=None
                img_evidence_mask=None      
                text_model_output=self.clipmodel.text_model(**claim_img_encod) 
                text_embeds=text_model_output.last_hidden_state 
                text_embeds=text_embeds[:,1:]
                claim_img_attention_masks=claim_img_attention_masks[:,1:]

        
        claim_embed= text_embeds[0:1]        #1,sequence size-1, 512.  remove CLS token
        claim_mask= claim_img_attention_masks[0:1]
        txt_evidences_embed= text_embeds[1:]    #n,sequence size, 512
        txt_evidence_mask=claim_img_attention_masks[1:] #TODO remove last SEP token     
        output=self.verifier(self.has_evidence(has_image),claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask )
        return output
    
    def has_evidence(self,has_image):
        if not self.model_type_enum.use_text_evidence and (not  self.model_type_enum.use_image_evidence or not has_image):
            return False
        else:
            return True
        
class Verifier(torch.nn.Module):        
    def __init__(self, max_image_num,model_type,model_type_enum,only_claim_setting,token_pooling_method ):    
        super(Verifier, self).__init__()
        self.only_claim_setting=only_claim_setting 
        if model_type_enum.stance_layer=="2":
            self.stance_detect_layer=StanceDetectionLayer2(512,max_image_num,model_type,model_type_enum)
        elif model_type_enum.stance_layer=="3":
            self.stance_detect_layer=StanceDetectionLayer3(512,max_image_num,model_type,model_type_enum)
        elif model_type_enum.stance_layer=="4": 
            self.stance_detect_layer= BertMultiwayMatch( 768,512,model_type_enum)
        elif model_type_enum.stance_layer=="5": 
            self.stance_detect_layer= BertMultiwayMatch2( 768,512,model_type_enum)
        elif model_type_enum.stance_layer=="6": 
            self.stance_detect_layer= BertMultiwayMatch3( 768,512,model_type_enum)    
        else:
            self.stance_detect_layer=StanceDetectionLayer(512,max_image_num,model_type,model_type_enum)
       
        if model_type_enum.sentiment_layer =="4":
            self.sentiment_detect_layer=SentimentDetectionLayer4(512,token_pooling_method)
        elif model_type_enum.sentiment_layer =="2":
            self.sentiment_detect_layer=SentimentDetectionLayer2(512,token_pooling_method)
        elif model_type_enum.sentiment_layer =="5":
            self.sentiment_detect_layer=SentimentDetectionLayer5(512,token_pooling_method)
        else:
            self.sentiment_detect_layer=SentimentDetectionLayer(512,token_pooling_method)
            
            
    def forward(self,   has_evidence,claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask):
        if has_evidence:
            output=self.stance_detect_layer(claim_embed,txt_evidences_embed,img_evidences_embed,has_image,claim_mask,txt_evidence_mask,img_evidence_mask)
        elif self.only_claim_setting=="stance":
            output=self.stance_detect_layer(claim_embed,claim_embed,None,False,claim_mask,claim_mask,None)
        else:
            output=self.sentiment_detect_layer(claim_embed  )
        return output
    
    