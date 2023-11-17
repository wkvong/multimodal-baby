#!/bin/bash

# cvc
# python alignment.py --model cvc --seed 0
# python alignment.py --model cvc --seed 1
# python alignment.py --model cvc --seed 2

# cvc (random features)
python alignment.py --model cvc_random_features --seed 0
python alignment.py --model cvc_random_features --seed 1
python alignment.py --model cvc_random_features --seed 2

# cvc (shuffled)
python alignment.py --model cvc_shuffled --seed 0
python alignment.py --model cvc_shuffled --seed 1
python alignment.py --model cvc_shuffled --seed 2

# clip (only one seed)
python alignment.py --model clip --seed 0

