#!/bin/bash
learning_rate=5e-4
batch_size=64
input_size=7500
embedding_size=128
margin=1.0
max_epochs=100
seed=0

# rnn layer (random init)
python train.py --learning_rate $learning_rate --batch_size $batch_size --input_size $input_size --embedding_size $embedding_size --margin $margin --max_epochs $max_epochs --seed $seed --lang_encoder rnn --exp_name multimodal_rnn_random_init

# # rnn layer (pretrained init)
python train.py --learning_rate $learning_rate --batch_size $batch_size --input_size $input_size --embedding_size $embedding_size --margin $margin --max_epochs $max_epochs --seed $seed --lang_encoder rnn --use_pretrained_lang --exp_name multimodal_rnn_pretrained_init

# word embedding layer only
python train.py --learning_rate $learning_rate --batch_size $batch_size --input_size $input_size --embedding_size $embedding_size --margin $margin --max_epochs $max_epochs --seed $seed --lang_encoder word_embed --exp_name multimodal_word_embed_random_init
