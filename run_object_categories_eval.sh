# object categories evaluation script

# evaluations for embedding model
python object_categories_eval.py --model embedding --seed 0 --eval_type image --save_predictions
python object_categories_eval.py --model embedding --seed 1 --eval_type image --save_predictions
python object_categories_eval.py --model embedding --seed 2 --eval_type image --save_predictions

python object_categories_eval.py --model embedding --seed 0 --eval_type text --save_predictions
python object_categories_eval.py --model embedding --seed 1 --eval_type text --save_predictions
python object_categories_eval.py --model embedding --seed 2 --eval_type text --save_predictions

# evaluation for lstm model
python object_categories_eval.py --model lstm --seed 0 --eval_type image --save_predictions
python object_categories_eval.py --model lstm --seed 1 --eval_type image --save_predictions
python object_categories_eval.py --model lstm --seed 2 --eval_type image --save_predictions

python object_categories_eval.py --model lstm --seed 0 --eval_type text --save_predictions
python object_categories_eval.py --model lstm --seed 1 --eval_type text --save_predictions
python object_categories_eval.py --model lstm --seed 2 --eval_type text --save_predictions



