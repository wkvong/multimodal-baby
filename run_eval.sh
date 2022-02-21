# evaluation for embedding model
python eval.py --use_kitty_label --model embedding --dataset dev --save_predictions

# evaluation for lstm model
python eval.py --use_kitty_label --model lstm --dataset dev --eval_include_sos_eos --save_predictions
