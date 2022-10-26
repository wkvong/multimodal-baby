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
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from multimodal.multimodal_data_module import EVAL_DATA_DIR, SOS_TOKEN_ID, EOS_TOKEN_ID, load_data
from multimodal.multimodal_lit import MultiModalLitModel
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # load our model checkpoint
    checkpoint_name = "multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_0"
    linear_probe_checkpoint_name = f"/home/wv9/code/WaiKeen/multimodal-baby/probe_results/{args.checkpoint}.tar"
    checkpoint = glob.glob(f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]

    # get config
    config = {}
    if 'seed_0' in args.checkpoint:
        config['seed'] = 0
    elif 'seed_1' in args.checkpoint:
        config['seed'] = 1
    elif 'seed_2' in args.checkpoint:
        config['seed'] = 2

    if "frozen_pretrained" in args.checkpoint:
        config["cnn"] = "frozen_pretrained"
    elif "finetuned_pretrained" in args.checkpoint:
        config["cnn"] = "finetuned_pretrained"

    if "subset_10" in args.checkpoint:
        config["model"] = "embedding_linear_probe_10_percent"
    elif "subset_1" in args.checkpoint:
        config["model"] = "embedding_linear_probe_1_percent"
    else:
        config["model"] = "embedding_linear_probe"
     
    # initialize linear probe model
    model = MultiModalLitModel.load_from_checkpoint(
            checkpoint, map_location=device)
    vision_model = model.vision_encoder
     
    # define custom vision model so that only first return arg from forward pass is used
    # which contains the actual embedding
    class VisionModelWrapper(nn.Module):
        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model
     
        def forward(self, x):
            x, y = self.vision_model(x)
            return x
     
    vision_model = VisionModelWrapper(vision_model)
    classifier = torch.nn.Linear(in_features=512, out_features=22, bias=True).to(device)
    model = torch.nn.Sequential(vision_model, classifier)
     
    # load linear probe
    model.load_state_dict(torch.load(linear_probe_checkpoint_name)['model_state_dict'], strict=False)
     
    # build data module
    data_module = MultiModalSAYCamDataModule()
    data_module.prepare_data()
    data_module.setup()
     
    # load vocab and metadata
    vocab = data_module.read_vocab()
    eval_data = load_data(EVAL_DATA_DIR / "eval_test.json")
    classes = sorted(os.listdir(EVAL_DATA_DIR / "eval" / "dev"))
     
    # initialize correct and total pred counts
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
     
    # create dataloader
    dataloader = {
        "dev": data_module.val_dataloader,
        "test": data_module.test_dataloader,
        }["test"]()[1]  # second dataloader contains eval data
     
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
            # "model": "linear_probe_embedding",
            "model": config["model"],
            "cnn": config["cnn"],
            "seed": config["seed"],
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
        results_filename = f"results/{config['model']}_{config['cnn']}_seed_{config['seed']}_image_saycam_eval_predictions.json"
     
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
