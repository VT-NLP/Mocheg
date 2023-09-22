#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.


import torch
from controllable.classification.train_verify import load_classifier
from controllable.generation.train_loop import * 

 
    

def main():

    model_args, data_args, training_args=init()
    # Detecting last checkpoint.
    last_checkpoint=get_checkpoint(training_args)
    tokenizer,model=load_model_tokenizer(model_args,data_args)
    train_dataset,eval_dataset,predict_dataset,raw_datasets,data_collator=load_data(model_args, data_args, training_args,tokenizer,model)
    # Metric
    metric = load_metric(data_args.metric)
     
    truthfulness_classifier,classifier_tokenizer=load_classifier(model_args.classifier_checkpoint_dir )
    logger.info(data_args.metric)
    
    def compute_metrics(eval_preds):
        return compute_metrics_with_parameter(eval_preds,tokenizer,data_args,metric)

    if data_args.is_compute_metrics_by_input !="y":
        trainer=get_trainer(model,training_args,train_dataset,eval_dataset,tokenizer,data_collator,
                            compute_metrics,truthfulness_classifier,training_args.control ,data_args.ignore_pad_token_for_loss,
                            classifier_tokenizer,data_args.max_target_length)
        
        train(training_args,trainer,data_args,last_checkpoint,train_dataset)

        max_length,num_beams=evaluate(training_args,trainer,eval_dataset,data_args)
        
        results=predict(training_args,trainer,predict_dataset,max_length,num_beams,data_args,tokenizer)
        
        try:
            create_model_card(model_args,data_args,trainer,training_args)
        except Exception as e:
            print("Fail to create_model_card. But it is optional:"+e)
        
        return results
    else:
        compute_metrics_by_input(raw_datasets,data_args,metric,data_args.metric)
        return
    
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


# def check_evidence():
#     preds = 
#     labels = 
#     compute_metrics(preds, labels)

if __name__ == "__main__":
    main()