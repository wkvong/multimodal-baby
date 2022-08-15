# image eval_shuffleduations for embedding model
python eval_shuffled.py --model embedding --seed 0 --stage dev --eval_type image --use_kitty_label --save_predictions
python eval_shuffled.py --model embedding --seed 1 --stage dev --eval_type image --use_kitty_label --save_predictions
python eval_shuffled.py --model embedding --seed 2 --stage dev --eval_type image --use_kitty_label --save_predictions

python eval_shuffled.py --model embedding --seed 0 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 
python eval_shuffled.py --model embedding --seed 1 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 
python eval_shuffled.py --model embedding --seed 2 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 

python eval_shuffled.py --model embedding --seed 0 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
python eval_shuffled.py --model embedding --seed 1 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
python eval_shuffled.py --model embedding --seed 2 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 

python eval_shuffled.py --model embedding --seed 0 --stage dev --eval_type text --use_kitty_label --save_predictions
python eval_shuffled.py --model embedding --seed 1 --stage dev --eval_type text --use_kitty_label --save_predictions
python eval_shuffled.py --model embedding --seed 2 --stage dev --eval_type text --use_kitty_label --save_predictions

python eval_shuffled.py --model embedding --seed 0 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 
python eval_shuffled.py --model embedding --seed 1 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 
python eval_shuffled.py --model embedding --seed 2 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 

python eval_shuffled.py --model embedding --seed 0 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
python eval_shuffled.py --model embedding --seed 1 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
python eval_shuffled.py --model embedding --seed 2 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 

# evaluation for lstm model
python eval_shuffled.py --model lstm --seed 0 --stage dev --eval_type image --use_kitty_label --save_predictions
python eval_shuffled.py --model lstm --seed 1 --stage dev --eval_type image --use_kitty_label --save_predictions
python eval_shuffled.py --model lstm --seed 2 --stage dev --eval_type image --use_kitty_label --save_predictions

python eval_shuffled.py --model lstm --seed 0 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 
python eval_shuffled.py --model lstm --seed 1 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 
python eval_shuffled.py --model lstm --seed 2 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 

python eval_shuffled.py --model lstm --seed 0 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
python eval_shuffled.py --model lstm --seed 1 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
python eval_shuffled.py --model lstm --seed 2 --stage dev --eval_type image --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 

python eval_shuffled.py --model lstm --seed 0 --stage dev --eval_type text --use_kitty_label --save_predictions
python eval_shuffled.py --model lstm --seed 1 --stage dev --eval_type text --use_kitty_label --save_predictions
python eval_shuffled.py --model lstm --seed 2 --stage dev --eval_type text --use_kitty_label --save_predictions

python eval_shuffled.py --model lstm --seed 0 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 
python eval_shuffled.py --model lstm --seed 1 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 
python eval_shuffled.py --model lstm --seed 2 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_9_foils.json 

python eval_shuffled.py --model lstm --seed 0 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
python eval_shuffled.py --model lstm --seed 1 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
python eval_shuffled.py --model lstm --seed 2 --stage dev --eval_type text --use_kitty_label --save_predictions --eval_metadata_filename eval_filtered_dev_21_foils.json 
