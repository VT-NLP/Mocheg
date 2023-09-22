### evidence retrieval
python retrieval/utils/preprocess.py
python retrieve_train.py --mode=test --train_config=IMAGE_MODEL --model_name=checkpoint/image_retrieval
python retrieve_train.py --mode=test --train_config=BI_ENCODER --model_name=checkpoint/text_retrieval/bi_encoder


### claim verification
python main.py --mode=preprocess_for_verification
python verify.py --mode=inference   --model_type=CLAIM_TEXT_IMAGE_attention_5_4 --batch_size=2048 --loss_weight_power=3  --checkpoint_dir=checkpoint/claim_verification/text_and_image_evidence


### explanation generation
python main.py --mode=preprocess_for_generation
CUDA_VISIBLE_DEVICES=0 python controllable_generation.py --do_predict --per_device_train_batch_size 8 --per_device_eval_batch_size 8  --predict_with_generate --text_column truth_claim_evidence --summary_column cleaned_ruling_outline  --model_name_or_path #(your_path_to_generation_model, e.g., checkpoint/controllable_generation/generation/without)  --test_file #(your_path_to_preprocessed_data, e.g., data/test/Corpus2_for_controllable_generation.csv)  --train_file #(your_path_to_preprocessed_data, e.g., data/train/Corpus2_for_controllable_generation.csv)   --validation_file #(your_path_to_preprocessed_data, e.g., data/val/Corpus2_for_controllable_generation.csv)   --output_dir controllable/generation/output/bart/run_0/without --classifier_checkpoint_dir=#(your_path_to_explanation_classify_model, e.g., checkpoint/controllable_generation/explanation_classify)
                       
