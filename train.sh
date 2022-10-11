### evidence retrieval
## train
python retrieval/utils/preprocess.py
python retrieve_train.py
python retrieve_train.py  --train_config=BI_ENCODER
## inference
python retrieve_train.py --mode=test --train_config=CROSS_ENCODER 
python retrieve_similarity_recall.py --bi_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00102-train_bi-encoder-multi-qa-MiniLM-L6-cos-v1-2022-08-06_12-55-10 --image_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00108-train_bi-encoder-clip-ViT-B-32-2022-08-06_18-30-14 --media=txt --no_rerank --top_k=5 --csv_out_dir=/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/retrieval/retrieval_result_no_rerank.csv
python retrieve_similarity_recall.py --bi_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00102-train_bi-encoder-multi-qa-MiniLM-L6-cos-v1-2022-08-06_12-55-10 --image_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00108-train_bi-encoder-clip-ViT-B-32-2022-08-06_18-30-14 --media=img_txt  --top_k=10 --csv_out_dir=/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/retrieval/retrieval_result_10.csv
python retrieve_similarity_recall.py --bi_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00102-train_bi-encoder-multi-qa-MiniLM-L6-cos-v1-2022-08-06_12-55-10 --image_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00108-train_bi-encoder-clip-ViT-B-32-2022-08-06_18-30-14 --media=txt --no_rerank --top_k=10 --csv_out_dir=/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/test/retrieval/retrieval_result_no_rerank_10.csv
python retrieve_similarity_recall.py --bi_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00102-train_bi-encoder-multi-qa-MiniLM-L6-cos-v1-2022-08-06_12-55-10 --image_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00108-train_bi-encoder-clip-ViT-B-32-2022-08-06_18-30-14 --media=img_txt  --top_k=5  
## generate output to support claim verification
python retrieve_similarity_recall.py --bi_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00102-train_bi-encoder-multi-qa-MiniLM-L6-cos-v1-2022-08-06_12-55-10 --image_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00108-train_bi-encoder-clip-ViT-B-32-2022-08-06_18-30-14 --media=img_txt  --top_k=5  --in_dir=/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/val --skip_score
python retrieve_similarity_recall.py --bi_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00102-train_bi-encoder-multi-qa-MiniLM-L6-cos-v1-2022-08-06_12-55-10 --image_encoder_checkpoint=/home/menglong/workspace/code/misinformation_detection/retrieval/output/runs_3/00108-train_bi-encoder-clip-ViT-B-32-2022-08-06_18-30-14 --media=img_txt  --top_k=5  --in_dir=/home/menglong/workspace/code/referred/conll2019-snopes-crawling/final_corpus/mocheg2/train --skip_score

### claim verification 
## train with gold evidence
python main.py --mode=preprocess_for_verification 
python verify.py     --model_type=CLAIM_TEXT_IMAGE_attention_5_4 --batch_size=2048 --lr=0.001 --loss_weight_power=2   
##generate verification_result.csv to support explanation generation
CUDA_VISIBLE_DEVICES=2 python verify.py --mode=inference --save_in_checkpoint_dir="n" --model_type=CLAIM_TEXT_IMAGE_attention_5_4   --checkpoint_dir=/home/menglong/workspace/code/misinformation_detection/verification/output/runs/00810- 
CUDA_VISIBLE_DEVICES=3 python verify.py --mode=inference --save_in_checkpoint_dir="n" --model_type=CLAIM_TEXT_IMAGE_attention_5_4   --checkpoint_dir=/home/menglong/workspace/code/misinformation_detection/verification/output/runs/00907- --evidence_file_name=retrieval/retrieval_result.csv 
 


### explanation generation
## train
python main.py --mode=preprocess_for_generation 
python main.py --mode=preprocess_for_generation_inference 
CUDA_VISIBLE_DEVICES=1 python explanation_classify.py 
# CUDA_VISIBLE_DEVICES=3  python controllable_generation.py    --do_train   --do_predict   --predict_with_generate --per_device_train_batch_size 10 --per_device_eval_batch_size 10   --classifier_checkpoint_dir=/home/menglong/workspace/code/misinformation_detection/controllable/classification/output/runs/00011- --num_train_epochs=30 --save_strategy=epoch --evaluation_strategy=epoch --logging_strategy=epoch --load_best_model_at_end --metric_for_best_model=eval_rouge1 --output_dir controllable/generation/output/bart/run_16/without  
CUDA_VISIBLE_DEVICES=0   python controllable_generation.py    --do_train    --do_predict    --per_device_train_batch_size 12 --per_device_eval_batch_size 12   --classifier_checkpoint_dir=/home/menglong/workspace/code/misinformation_detection/controllable/classification/output/runs/00011- --num_train_epochs=40 --output_dir controllable/generation/output/bart/run_28/without --gradient_accumulation_steps=16 --learning_rate=0.0001  --predict_with_generate  --sc_weight=0.5  

 