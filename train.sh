### evidence retrieval
python retrieval/utils/preprocess.py
python retrieve_train.py
python retrieve_train.py  --train_config=BI_ENCODER
python retrieve_similarity_recall.py --bi_encoder_checkpoint=checkpoint/text_retrieval/bi_encoder  --image_encoder_checkpoint=checkpoint/image_retrieval

### claim verification
python main.py --mode=preprocess_for_verification
python verify.py --model_type=CLAIM_TEXT_IMAGE_attention_5_4 --batch_size=4 --lr=0.01 --loss_weight_power=3  