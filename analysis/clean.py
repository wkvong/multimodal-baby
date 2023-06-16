# script to gather evaluation results

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from siuba import group_by, summarize, arrange, filter, mutate, if_else, _
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

import torch
from torchvision import transforms
from multimodal.multimodal_lit import MultiModalLitModel
import clip
from sklearn.manifold import TSNE
from PIL import Image

# saycam labeled s results
embedding_results = ["../results/saycam/embedding_frozen_pretrained_seed_0_image_saycam_test_eval_predictions.json",
"../results/saycam/embedding_frozen_pretrained_seed_1_image_saycam_test_eval_predictions.json",
"../results/saycam/embedding_frozen_pretrained_seed_2_image_saycam_test_eval_predictions.json"]
# embedding_filtered_results = ["../results/saycam/embedding_frozen_pretrained_seed_0_image_saycam_test_eval_filtered_predictions.json",
# "../results/saycam/embedding_frozen_pretrained_seed_1_image_saycam_test_eval_filtered_predictions.json",
# "../results/saycam/embedding_frozen_pretrained_seed_2_image_saycam_test_eval_filtered_predictions.json"]
embedding_manual_filtered_results = ["../results/saycam/embedding_frozen_pretrained_seed_0_image_saycam_test_eval_manual_filtered_predictions.json",
"../results/saycam/embedding_frozen_pretrained_seed_1_image_saycam_test_eval_manual_filtered_predictions.json",
"../results/saycam/embedding_frozen_pretrained_seed_2_image_saycam_test_eval_manual_filtered_predictions.json"]

shuffled_results = ["../results/saycam/shuffle_embedding_frozen_pretrained_seed_0_image_saycam_test_eval_predictions.json",
"../results/saycam/shuffle_embedding_frozen_pretrained_seed_1_image_saycam_test_eval_predictions.json",
"../results/saycam/shuffle_embedding_frozen_pretrained_seed_2_image_saycam_test_eval_predictions.json"]
clip_results = ["../results/saycam/clip_image_saycam_test_eval_predictions.json"]

frozen_linear_probe_all_results = [
    "../results/saycam/embedding_linear_probe_seed_0_image_saycam_eval_predictions.json",
    "../results/saycam/embedding_linear_probe_seed_1_image_saycam_eval_predictions.json",
    "../results/saycam/embedding_linear_probe_seed_2_image_saycam_eval_predictions.json"
]
frozen_linear_probe_10_percent_results = [
    "../results/saycam/embedding_linear_probe_10_percent_seed_0_image_saycam_eval_predictions.json",
    "../results/saycam/embedding_linear_probe_10_percent_seed_1_image_saycam_eval_predictions.json",
    "../results/saycam/embedding_linear_probe_10_percent_seed_2_image_saycam_eval_predictions.json"
]
frozen_linear_probe_1_percent_results = [
    "../results/saycam/embedding_linear_probe_1_percent_seed_0_image_saycam_eval_predictions.json",
    "../results/saycam/embedding_linear_probe_1_percent_seed_1_image_saycam_eval_predictions.json",
    "../results/saycam/embedding_linear_probe_1_percent_seed_2_image_saycam_eval_predictions.json"
]

saycam_results = []

for results in embedding_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_embedding"
    result_df["filtered"] = False
    saycam_results.append(result_df)

# for results in embedding_filtered_results:
#     with open(results) as f:
#         data = json.load(f)

#     result_df = pd.DataFrame(data["data"])
#     result_df["target_category"] = result_df["categories"].str[0]
    
#     # add extra columns
#     result_df["config"] = "contrastive_embedding"
#     result_df["filtered"] = True
#     saycam_results.append(result_df)

for results in embedding_manual_filtered_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_embedding"
    result_df["filtered"] = True
    saycam_results.append(result_df)
    
for results in shuffled_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_shuffled"
    saycam_results.append(result_df)
    
for results in frozen_linear_probe_all_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "supervised_linear_probe_all"
    saycam_results.append(result_df)
    
for results in frozen_linear_probe_1_percent_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "supervised_linear_probe_1_percent"
    saycam_results.append(result_df)
    
for results in frozen_linear_probe_10_percent_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "supervised_linear_probe_10_percent"
    saycam_results.append(result_df)
    
for results in clip_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "clip (vit-l/14)"
    result_df["seed"] = 0
    saycam_results.append(result_df)
    
# combine results
saycam_results_df = pd.concat(saycam_results)

# save results
print("saving saycam bounds results to csv")
saycam_results_df.to_csv("../results/summary/saycam-bounds-summary.csv")

saycam_ablations = []
embedding_results = ["../results/saycam/embedding_frozen_pretrained_seed_0_image_saycam_test_eval_predictions.json",
"../results/saycam/embedding_frozen_pretrained_seed_1_image_saycam_test_eval_predictions.json",
"../results/saycam/embedding_frozen_pretrained_seed_2_image_saycam_test_eval_predictions.json"]
lstm_results = ["../results/saycam/lstm_frozen_pretrained_seed_0_image_saycam_test_eval_predictions.json",
"../results/saycam/lstm_frozen_pretrained_seed_1_image_saycam_test_eval_predictions.json",
"../results/saycam/lstm_frozen_pretrained_seed_2_image_saycam_test_eval_predictions.json"]
# embedding_finetune_pretrained_init_results = ["../results/saycam/embedding_finetune_pretrained_seed_0_image_saycam_test_eval_predictions.json",
# "../results/saycam/embedding_finetune_pretrained_seed_1_image_saycam_test_eval_predictions.json",
# "../results/saycam/embedding_finetune_pretrained_seed_2_image_saycam_test_eval_predictions.json"]
embedding_finetune_random_init_results = ["../results/saycam/embedding_finetune_random_init_seed_0_image_saycam_test_eval_predictions.json", 
"../results/saycam/embedding_finetune_random_init_seed_1_image_saycam_test_eval_predictions.json",
"../results/saycam/embedding_finetune_random_init_seed_2_image_saycam_test_eval_predictions.json"]
embedding_frozen_random_init_results = ["../results/saycam/embedding_frozen_random_init_seed_0_image_saycam_test_eval_predictions.json", 
"../results/saycam/embedding_frozen_random_init_seed_1_image_saycam_test_eval_predictions.json",
"../results/saycam/embedding_frozen_random_init_seed_2_image_saycam_test_eval_predictions.json"]
single_frame_results = ["../results/saycam/embedding_frozen_pretrained_multiple_frames_False_seed_0_image_saycam_test_eval_predictions.json",
"../results/saycam/embedding_frozen_pretrained_multiple_frames_False_seed_1_image_saycam_test_eval_predictions.json",
"../results/saycam/embedding_frozen_pretrained_multiple_frames_False_seed_2_image_saycam_test_eval_predictions.json"]
# no_data_aug_results = ["../results/saycam/embedding_frozen_pretrained_augment_frames_False_seed_0_image_saycam_test_eval_predictions.json",
# "../results/saycam/embedding_frozen_pretrained_augment_frames_False_seed_1_image_saycam_test_eval_predictions.json",
# "../results/saycam/embedding_frozen_pretrained_augment_frames_False_seed_2_image_saycam_test_eval_predictions.json"]

for results in embedding_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_embedding"
    saycam_ablations.append(result_df)

for results in lstm_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_lstm"
    saycam_ablations.append(result_df)
    
# for results in embedding_finetune_pretrained_init_results:
#     with open(results) as f:
#         data = json.load(f)

#     result_df = pd.DataFrame(data["data"])
#     result_df["target_category"] = result_df["categories"].str[0]
    
#     # add extra columns
#     result_df["config"] = "contrastive_embedding_finetune_pretrained_init"
#     saycam_ablations.append(result_df)
    
for results in embedding_finetune_random_init_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_embedding_finetune_random_init"
    saycam_ablations.append(result_df)

for results in embedding_frozen_random_init_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_embedding_frozen_random_init"
    saycam_ablations.append(result_df)
    
for results in single_frame_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_embedding_single_frame"
    saycam_ablations.append(result_df)
    
# for results in no_data_aug_results:
#     with open(results) as f:
#         data = json.load(f)

#     result_df = pd.DataFrame(data["data"])
#     result_df["target_category"] = result_df["categories"].str[0]
    
#     # add extra columns
#     result_df["config"] = "contrastive_embedding_no_data_aug"
#     saycam_ablations.append(result_df)
    
saycam_ablations = pd.concat(saycam_ablations)

# save results
print("saving saycam ablation results to csv")
saycam_ablations.to_csv("../results/summary/saycam-ablations.csv", index=False)

# object categories
object_categories_embedding_results = ["../results/object_categories/embedding_frozen_pretrained_seed_0_image_object_categories_test_eval_predictions.json",
"../results/object_categories/embedding_frozen_pretrained_seed_1_image_object_categories_test_eval_predictions.json",
"../results/object_categories/embedding_frozen_pretrained_seed_2_image_object_categories_test_eval_predictions.json"]

object_categories_frozen_random_init_embedding_results = ["../results/object_categories/embedding_frozen_random_init_seed_0_image_object_categories_test_eval_predictions.json",
"../results/object_categories/embedding_frozen_random_init_seed_1_image_object_categories_test_eval_predictions.json",
"../results/object_categories/embedding_frozen_random_init_seed_2_image_object_categories_test_eval_predictions.json"]

object_categories_shuffled_embedding_results = ["../results/object_categories/shuffle_embedding_frozen_pretrained_seed_0_image_object_categories_test_eval_predictions.json",
"../results/object_categories/shuffle_embedding_frozen_pretrained_seed_1_image_object_categories_test_eval_predictions.json",
"../results/object_categories/shuffle_embedding_frozen_pretrained_seed_2_image_object_categories_test_eval_predictions.json"]

object_categories_clip_results = ["../results/object_categories/clip_image_object_categories_test_eval_predictions.json"]

object_categories_linear_probe_results = ["../results/object_categories/embedding_object_categories_linear_probe_seed_0_split_first_image_object_categories_eval_predictions.json", "../results/object_categories/embedding_object_categories_linear_probe_seed_0_split_last_image_object_categories_eval_predictions.json",
"../results/object_categories/embedding_object_categories_linear_probe_seed_1_split_first_image_object_categories_eval_predictions.json", "../results/object_categories/embedding_object_categories_linear_probe_seed_1_split_last_image_object_categories_eval_predictions.json",
"../results/object_categories/embedding_object_categories_linear_probe_seed_2_split_first_image_object_categories_eval_predictions.json",
"../results/object_categories/embedding_object_categories_linear_probe_seed_2_split_last_image_object_categories_eval_predictions.json"]

object_categories_results = []

for results in object_categories_embedding_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive"
    result_df["split"] = None
    object_categories_results.append(result_df)

for results in object_categories_frozen_random_init_embedding_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_frozen_random_init"
    result_df["split"] = None
    object_categories_results.append(result_df)

for results in object_categories_shuffled_embedding_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["split"] = None
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "contrastive_shuffled"
    result_df["split"] = None
    object_categories_results.append(result_df)

for results in object_categories_clip_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["split"] = None
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "clip"
    result_df["split"] = None
    object_categories_results.append(result_df)
    
for results in object_categories_linear_probe_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]
    
    # add extra columns
    result_df["config"] = "linear_probe"
    object_categories_results.append(result_df)
    

# combine results
object_categories_results_df = pd.concat(object_categories_results)

# save as csv
print("saving object categories results to csv")
object_categories_results_df.to_csv("../results/summary/object-categories.csv")

