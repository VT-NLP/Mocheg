import torch.nn as nn
 
import torch.nn.functional as F
import torch 
def show_model(model):
    for p in model.named_parameters():
      
        print(p[0], p[1].requires_grad)
        
        
def freeze_part_bert(bert_model,freeze_layer_num):
    count = 0
    for p in bert_model.named_parameters():
        
        if (count<=freeze_layer_num):
            p[1].requires_grad=False    
        else:
            break
              
        count=count+1
        print(p[0], p[1].requires_grad)
        
    return bert_model 

class TextModel(torch.nn.Module):
        
    def __init__(self, bert, concat_hidden_size, final_hidden_size, num_classes,freeze_bert_layer_number ):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TextModel, self).__init__()
        freeze_part_bert(bert,freeze_bert_layer_number)
        self.bert = bert
        
        
        
        self.fcbert1 = nn.Linear(768, 128)
        self.fcbert2 = nn.Linear(128, 16)
        self.fcbert3 = nn.Linear(16, 3)
        
        
    #         self.fc1 = nn.Linear(8960, concat_hidden_size)
        
    #         self.fc2 = nn.Linear(concat_hidden_size, final_hidden_size)
        
    #         self.fc3 = nn.Linear(26, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        


    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # 1, 768
        output_bert = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids).pooler_output
        
        
        
    #         sep_output = output_bert.last_hidden_state[:, 511]   #1, 256

        
    #         concat_cls_sep = torch.concat( (cls_output, sep_output), 1 )  #1, 512
        
        output = self.dropout(F.leaky_relu(self.fcbert1(output_bert), .1))    #1, 128
            
        output = self.dropout(F.leaky_relu(self.fcbert2(output), 0.1))     #1, 16
        
        output = self.fcbert3(output)    #1, 3
            
        return output