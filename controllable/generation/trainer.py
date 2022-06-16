from torch import nn
from transformers import Seq2SeqTrainer, Trainer
import torch
from typing import List, Optional, Tuple, Union
from controllable.generation.loss.style_loss import get_sc_loss 
from torch.utils.data import DataLoader
class CGSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, model  = None, args = None,
                 data_collator  = None, train_dataset = None, eval_dataset  = None,
                 tokenizer = None, model_init = None,
                 compute_metrics  = None, callbacks = None, 
                 optimizers  =  (None, None), 
                 preprocess_logits_for_metrics = None,truthfulness_classifier=None,control_list=None,
                 ignore_pad_token_for_loss=None,classifier_tokenizer=None,max_len=None):
        
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, 
                         compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.truthfulness_classifier=truthfulness_classifier
        self.classifier_device=truthfulness_classifier.bert.device
        self.control_list=control_list
        self.ignore_pad_token_for_loss=ignore_pad_token_for_loss
        self.classifier_tokenizer=classifier_tokenizer
        self.max_len=max_len
        self.loss_step=0
            
    
            
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs=super().compute_loss(model,inputs,True) #TODO can reuse the greedy search method to speed up
        model_device=self.model.device 
        classifier_device=self.truthfulness_classifier.bert.device
        truthfulness=inputs.get("truthfulness")
        logits = outputs.get("logits")
        labels=inputs.get("labels")
        if "truthfulness" in self.control_list and (20 < self.loss_step or len(self.train_dataset)< self.loss_step):
            if self.truthfulness_classifier.bert.device.index!=model_device.index:             
                labels=labels.to(classifier_device)
                logits=logits.to(classifier_device)
                truthfulness=truthfulness.to(classifier_device)
            loss_sc,_=get_sc_loss(labels,self.tokenizer,logits,self.truthfulness_classifier,truthfulness,model_device,
                              self.ignore_pad_token_for_loss,self.classifier_tokenizer,self.max_len)
            if self.truthfulness_classifier.bert.device.index!=model_device.index:   
                loss_sc=loss_sc.to(model_device)
            loss=loss+loss_sc*self.args.sc_weight
        self.loss_step+=1
        return (loss, outputs) if return_outputs else loss
    
    # def extra_metrics_for_test(self,predicts):
        # TODO logic for bleu
        
    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ):
    #     eval_loop_output=super().evaluation_loop(dataloader,description,prediction_loss_only,ignore_keys,metric_key_prefix)
        
    #     if metric_key_prefix =="test":
    #         predictions=eval_loop_output.all_preds
    #         label_ids=eval_loop_output.all_labels  
    #         metrics=eval_loop_output.metrics 
            
    #         if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #             if args.include_inputs_for_metrics:
    #                 metrics = self.compute_metrics(
    #                     EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
    #                 )
    #             else:
    #                 metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #         else:
    #             metrics = {}

    #         # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #         metrics = denumpify_detensorize(metrics)

        

    #         # Prefix all keys with metric_key_prefix + '_'
    #         for key in list(metrics.keys()):
    #             if not key.startswith(f"{metric_key_prefix}_"):
    #                 metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        