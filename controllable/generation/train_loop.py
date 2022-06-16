
from nltk import tokenize
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
from statistics import mean
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    # Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from controllable.generation.call_back import TimerCallback
# from controllable.generation.data.collator import DataCollatorForControllableSeq2Seq
from controllable.generation.model.controllable_bart import BartForControllableGeneration
from controllable.generation.trainer import CGSeq2SeqTrainer
from controllable.generation.util.argument import *
 
 
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

def init_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)
init_nltk()


def init():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CGSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Set seed before initializing model.
    set_seed(training_args.seed)
    return model_args, data_args, training_args

def get_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
   
    return last_checkpoint

# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
def get_dataset(data_args,model_args):
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    return raw_datasets

def load_model_tokenizer(model_args,data_args):
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
     
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.truncation_side="left"
    model = BartForControllableGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )
    return  tokenizer,model

def preprocess_dataset(training_args,data_args,raw_datasets):
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    return column_names,text_column,summary_column


def preprocess_function(examples,text_column,summary_column,prefix,data_args,padding,tokenizer,max_target_length):
    inputs = examples[text_column]
    targets = examples[summary_column]
    inputs = [prefix + inp for inp in inputs]
    truthfulness_list=examples["cleaned_truthfulness_int"]
    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["truthfulness"]=truthfulness_list
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metric_with_text_bertscore(decoded_preds,decoded_labels,is_compute_metrics_by_input,num_limit,metric):
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels,is_compute_metrics_by_input,num_limit)

    
    all_result = metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    avg_result=mean(all_result["f1"])
    result={"bertscore":avg_result}
    
    # Extract a few results from ROUGE
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()} TODO
    result = {key: value* 100 for key, value in result.items()} 
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def compute_metric_with_text_bleu(decoded_preds,decoded_labels,is_compute_metrics_by_input,num_limit,metric):
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels,is_compute_metrics_by_input,num_limit)
    decoded_preds_bleu  = [pred.split(' ') for pred in decoded_preds]
    decoded_labels_bleu = [[pred.split(' ')] for pred in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds_bleu, references=decoded_labels_bleu )
    bleu_value=result["bleu"]
    bleu_value=round(bleu_value* 100,4)
    result["bleu"]=bleu_value
    # Extract a few results from ROUGE
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()} TODO
    
    return result

def compute_metrics_with_text(decoded_preds,decoded_labels,is_compute_metrics_by_input,num_limit,metric,metric_name):
    if metric_name=="bleu":
        return compute_metric_with_text_bleu(decoded_preds,decoded_labels,is_compute_metrics_by_input,num_limit,metric)
    elif metric_name=="bertscore":#data_args.metric
        return compute_metric_with_text_bertscore(decoded_preds,decoded_labels,is_compute_metrics_by_input,num_limit,metric)
    else:
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels,is_compute_metrics_by_input,num_limit)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()} 
        result = {k: round(v, 4) for k, v in result.items()}
        return result
def compute_metrics_by_input(raw_datasets,data_args,metric,metric_name):
    preds=raw_datasets["test"]["Evidence"]
    labels=raw_datasets["test"]["cleaned_ruling_outline"] 
    
    cur_result=compute_metrics_with_text(preds,labels, data_args.is_compute_metrics_by_input, data_args.num_limit,metric,metric_name)
    # cur_result = {k: round(v, 4) for k, v in cur_result.items()} TODO
    print(cur_result)
        
        

def preprocess_one_subdataset(raw_datasets,subdataset_key,max_samples,max_target_length,training_args,data_args,column_names,
                              text_column,summary_column,prefix,padding,tokenizer ):
    if subdataset_key not in raw_datasets:
        raise ValueError(f"--do_{subdataset_key} requires a {subdataset_key} dataset")
    sub_dataset = raw_datasets[subdataset_key]
    if max_samples is not None:
        sub_dataset = sub_dataset.select(range(max_samples))
    with training_args.main_process_first(desc=f"{subdataset_key} dataset map pre-processing"):
        sub_dataset = sub_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Running tokenizer on {subdataset_key} dataset",
            fn_kwargs={"text_column":text_column,"summary_column":summary_column,"prefix":prefix,"data_args":data_args
                        ,"padding":padding,"tokenizer":tokenizer,"max_target_length":max_target_length}
        )
    return sub_dataset
        
def load_data(model_args, data_args, training_args,tokenizer,model):
    
    

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    raw_datasets=get_dataset(data_args,model_args)
    column_names,text_column,summary_column=preprocess_dataset(training_args,data_args,raw_datasets)

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


    if training_args.do_train:
        train_dataset=preprocess_one_subdataset(raw_datasets,"train",data_args.max_train_samples,max_target_length,training_args,
                                  data_args,column_names,text_column,summary_column,prefix,padding,tokenizer )        
    else:
        train_dataset=None
        

    if training_args.do_eval:
        eval_dataset=preprocess_one_subdataset(raw_datasets,"validation",data_args.max_eval_samples,data_args.val_max_target_length,training_args,
                                  data_args,column_names,text_column,summary_column,prefix,padding,tokenizer )   
    else:
        eval_dataset=None
        

    if training_args.do_predict:
        predict_dataset=preprocess_one_subdataset(raw_datasets,"test",data_args.max_predict_samples,data_args.val_max_target_length,training_args,
                                  data_args,column_names,text_column,summary_column,prefix,padding,tokenizer )   
    else:
        predict_dataset=None
         

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )        
    return train_dataset,eval_dataset,predict_dataset,raw_datasets,data_collator
def get_trainer(model,training_args,train_dataset,eval_dataset,tokenizer,data_collator,compute_metrics,truthfulness_classifier,
                control_list,ignore_pad_token_for_loss,classifier_tokenizer,max_len):
        # Initialize our Trainer
    trainer = CGSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        truthfulness_classifier=truthfulness_classifier,
        control_list=control_list,
        ignore_pad_token_for_loss=ignore_pad_token_for_loss,
        classifier_tokenizer=classifier_tokenizer,
        max_len=max_len,
        callbacks=[TimerCallback]
    )
    return trainer         

def train(training_args,trainer,data_args,last_checkpoint,train_dataset):
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
def evaluate(training_args,trainer,eval_dataset,data_args):
    # Evaluation
    
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    return max_length,num_beams

def predict(training_args,trainer,predict_dataset,max_length,num_beams,data_args,tokenizer):
    # father_dir=training_args.output_dir
    # os.makedirs(os.path.join(training_args.output_dir,"without"), exist_ok=True)
    # os.makedirs(os.path.join(training_args.output_dir,"after_retrieval"), exist_ok=True)
    # os.makedirs(os.path.join(training_args.output_dir,"after_verification"), exist_ok=True)
    # os.makedirs(os.path.join(training_args.output_dir,"after_retrieval_verification"), exist_ok=True)
    # prediction_file=os.path.join(run_dir,f"without/predict_{opt.style}.csv")
    # reference_file='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/Corpus2_for_controllable_generation_200.csv'
    # after_retrieval_prediction_file=os.path.join(run_dir,f"after_retrieval/predict_{opt.style}.csv") 
    # after_retrieval_reference_file='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/generation/Corpus2_for_controllable_generation_after_retrieval.csv'
    # after_verification_prediction_file=os.path.join(run_dir,f"after_verification/predict_{opt.style}.csv")  
    # after_verification_reference_file='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/generation/Corpus2_for_controllable_generation_after_verification.csv'
    # after_retrieval_verification_prediction_file=os.path.join(run_dir,f"after_retrieval_verification/predict_{opt.style}.csv") 
    # after_retrieval_verification_reference_file='/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/generation/Corpus2_for_controllable_generation_after_retrieval_verification.csv'

    predict_one_setting(training_args,trainer,predict_dataset,max_length,num_beams,data_args,tokenizer)

#TODO deepcopy trainer, tokenizer, load dataset. then you can mp
def predict_one_setting(training_args,trainer,predict_dataset,max_length,num_beams,data_args,tokenizer):
    results = {}
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        # trainer.log_metrics("predict", metrics)  TODO TypeError: unsupported format string passed to list.__format__
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
    return results

def create_model_card(model_args,data_args,trainer,training_args):
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def compute_metrics_with_parameter(eval_preds,tokenizer,data_args,metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result=compute_metrics_with_text(decoded_preds,decoded_labels,data_args.is_compute_metrics_by_input,data_args.num_limit,metric,data_args.metric )
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] =  round(np.mean(prediction_lens),4)
    print(result)
    return result


from nltk.tokenize.treebank import TreebankWordDetokenizer    
def postprocess_text(preds, labels,is_compute_metrics_by_input,args_num_limit):
    if is_compute_metrics_by_input =="y" :
        num_limit=args_num_limit
    else:
        num_limit=None
    if num_limit!=None:
        len_limited_preds=[]
        for pred in preds:
            tokens=tokenize.word_tokenize(pred)
            token_num=len(tokens)
            if token_num>num_limit:     
                new_tokens=tokens[:num_limit]
                
                pred=TreebankWordDetokenizer().detokenize(new_tokens)
            len_limited_preds.append(pred)
    else:
        len_limited_preds=preds
        
    preds = [pred.strip() for pred in len_limited_preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in len_limited_preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels