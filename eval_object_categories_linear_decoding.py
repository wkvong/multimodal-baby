# run evaluation on linear probes
import argparse
import glob
import json
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from multimodal.multimodal_data_module import EVAL_DATA_DIR, SOS_TOKEN_ID, EOS_TOKEN_ID, load_data
from multimodal.multimodal_lit import MultiModalLitModel
from multimodal.object_categories_data_module import ObjectCategoriesDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # load our model checkpoint
    linear_probe_checkpoint_name = f"/home/wv9/code/WaiKeen/multimodal-baby/probe_results/{args.checkpoint}.tar"

    # get config
    config = {}

    config["model"] = "embedding_object_categories_linear_probe"
    
    if "seed_0" in args.checkpoint:
        config["seed"] = 0
    elif "seed_1" in args.checkpoint:
        config["seed"] = 1
    elif "seed_2" in args.checkpoint:
        config["seed"] = 2

    if "split_first" in args.checkpoint:
        config["split"] = "first"
    elif "split_last" in args.checkpoint:
        config["split"] = "last"
        
    # initialize linear probe model
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = torch.nn.Linear(
        in_features=2048, out_features=64, bias=True)
    model = model.to(device)
    model.load_state_dict(torch.load(linear_probe_checkpoint_name)['model_state_dict'], strict=False)

    # set up args
    dm_args = argparse.Namespace(
        eval_type="image",
        clip_eval=False)

    # build data module    
    data_module = ObjectCategoriesDataModule(dm_args)
    data_module.prepare_data()
    data_module.setup()
     
    # load vocab and metadata
    vocab = data_module.read_vocab()
    eval_data = load_data(EVAL_DATA_DIR / "eval_object_categories.json")
    classes = sorted(os.listdir(EVAL_DATA_DIR / "object_categories"))
     
    # initialize correct and total pred counts
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
     
    # create dataloader
    dataloader = data_module.test_dataloader()
     
    # initialize results
    results = []
     
    for i, batch in enumerate(dataloader):
        imgs, labels, label_lens, raw_labels = batch
     
        # get text category label
        class_label = raw_labels[0][0]
        class_idx = classes.index(class_label)
     
        imgs = imgs.squeeze(0).to(device)
        outputs = model(imgs)
        logits_list = outputs[:, class_idx]
        pred = torch.argmax(logits_list).item()
        logits_list = logits_list.detach().cpu().numpy().tolist()
     
        ground_truth = 0
        correct = False
        if pred == ground_truth:
            correct = True
            correct_pred[class_label] += 1
     
        total_pred[class_label] += 1
     
        # get categories
        curr_trial = eval_data[i]
        curr_target_category = curr_trial["target_category"]
        curr_foil_categories = curr_trial["foil_categories"]
        curr_eval_categories = [curr_target_category] + curr_foil_categories
     
        # store results
        curr_results = {
            "checkpoint": args.checkpoint,
            "model": config["model"],
            "seed": config["seed"],
            "split": config["split"],
            "eval_type": "image",
            "eval_dataset": "saycam",
            "stage": "test",
            "trial_idx": i,
            "categories": curr_eval_categories,
            "logits": logits_list,
            "pred": pred,
            "correct": correct,
            }
        results.append(curr_results)
     
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = float(correct_count) / total_pred[classname]
        print(f"Accuracy for class {classname:8s} is: {accuracy:.1%}")
     
    # print total accuracy
    total_correct = sum(correct_pred.values())
    total = sum(total_pred.values())
    print(f"Total accuracy: {total_correct / total:%}")
     
    # save results
    if args.save_predictions:
        # put results into a dictionary
        results_dict = {"data": results}
     
        # create dir
        os.makedirs('results', exist_ok=True)
     
        # get filename
        results_filename = f"results/object_categories/{config['model']}_seed_{config['seed']}_split_{config['split']}_image_object_categories_eval_predictions.json"
     
        # save to JSON
        print(f"Saving predictions to {results_filename}")
        with open(results_filename, "w") as f:
            json.dump(results_dict, f)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation with linear probe models')
    parser.add_argument('--checkpoint', type=str, help="path to linear probe checkpoint")
    parser.add_argument("--save_predictions", action="store_true",
                        help="save model predictions to JSON")
    args = parser.parse_args()
    main(args)
