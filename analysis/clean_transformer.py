# script to gather evaluation results for transformer model

import json
import pandas as pd

# saycam labeled s results
transformer_results = ["../results/saycam/transformer_frozen_pretrained_seed_0_image_saycam_test_eval_predictions.json",
                       "../results/saycam/transformer_frozen_pretrained_seed_1_image_saycam_test_eval_predictions.json",
                       "../results/saycam/transformer_frozen_pretrained_seed_2_image_saycam_test_eval_predictions.json"]

saycam_results = []
for results in transformer_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]

    # add extra columns
    result_df["config"] = "contrastive_transformer_embedding"
    result_df["filtered"] = False
    saycam_results.append(result_df)

# combine results
saycam_results_df = pd.concat(saycam_results)

# save results
print("saving saycam transformer results to csv")
saycam_results_df.to_csv("../results/summary/saycam-transformer-summary.csv")

object_categories_transformer_results = ["../results/object_categories/transformer_frozen_pretrained_seed_0_image_object_categories_test_eval_predictions.json",
                                         "../results/object_categories/transformer_frozen_pretrained_seed_1_image_object_categories_test_eval_predictions.json",
                                         "../results/object_categories/transformer_frozen_pretrained_seed_2_image_object_categories_test_eval_predictions.json"]

object_categories_results = []
for results in object_categories_transformer_results:
    with open(results) as f:
        data = json.load(f)

    result_df = pd.DataFrame(data["data"])
    result_df["target_category"] = result_df["categories"].str[0]

    # add extra columns
    result_df["config"] = "contrastive_transformer_embedding"
    result_df["split"] = None
    object_categories_results.append(result_df)

# combine results
object_categories_results_df = pd.concat(object_categories_results)

# save results
print("saving object categories transformer results to csv")
object_categories_results_df.to_csv("../results/summary/object-categories-transformer-summary.csv")

