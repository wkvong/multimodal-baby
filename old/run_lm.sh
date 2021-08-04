# embedding size: 64, dropout: 0.1
python train_lm.py --hidden_size 64 --dropout_p 0.1 --exp_name lstm_lm_embedding_size_64_dropout

# embedding size: 64, dropout: 0
python train_lm.py --hidden_size 64 --dropout_p 0.0 --exp_name lstm_lm_embedding_size_64_no_dropout

# embedding size: 128, dropout: 0.1
python train_lm.py --hidden_size 128 --dropout_p 0.1 --exp_name lstm_lm_embedding_size_128_dropout

# embedding size: 128, dropout: 0
python train_lm.py --hidden_size 128 --dropout_p 0.0 --exp_name lstm_lm_embedding_size_128_no_dropout
