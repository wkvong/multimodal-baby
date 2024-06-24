# Evaluation and Visualization for "[A systematic investigation of learnability from single child linguistic input](https://cims.nyu.edu/~brenden/papers/QinEtAl2024CogSci.pdf)"

This code is derived from [Finding Structure in One Child's Linguistic Experience](https://cims.nyu.edu/~brenden/papers/WangEtAl2023CognitiveScience.pdf).

## preparation
Provide paths to model checkpoints in [analysis_tools/checkpoints.py](analysis_tools/checkpoints.py).

## t-SNE and dendrogram visualization
Run relevant parts of [notebooks/lm_clustering.ipynb](notebooks/lm_clustering.ipynb).

## evaluation on the cloze test
Run [pos_analysis.py](pos_analysis.py), which is extracted from [notebooks/pos_analysis.ipynb](notebooks/pos_analysis.ipynb).
