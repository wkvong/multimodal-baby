import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from multimodal.multimodal_data_module import EVAL_DATA_DIR, SOS_TOKEN_ID, EOS_TOKEN_ID, load_data
from multimodal.object_categories_data_module import ObjectCategoriesDataModule, _get_vocab, _get_object_categories
from multimodal.multimodal import MultiModalModel
from multimodal.multimodal_lit import MultiModalLitModel
from train import _setup_parser

import clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # get checkpoint
    if args.model == "clip":
        print("Loading CLIP")
        checkpoint_name = "clip_vitb16"
        model, preprocess = clip.load("ViT-B/16", device=device)
        model.eval()

        # set up parser
        parser = _setup_parser()
        data_args = parser.parse_args("")
    else:
        if args.model == "embedding":
            # checkpoint_name = "multimodal_text_encoder_embedding_lr_0.0001_weight_decay_0.1_fix_temperature_True_batch_size_16"
            checkpoint_name = f"multimodal_text_encoder_embedding_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_{args.seed}"
        elif args.model == "lstm":
            # checkpoint_name = 'multimodal_text_encoder_lstm_lr_0.0001_weight_decay_0.2_fix_temperature_False_batch_size_8'
            # checkpoint_name = "multimodal_text_encoder_lstm_embedding_dim_512_fix_temperature_True_temperature_0.07_batch_size_8_dropout_i_0.5_lr_5e-05_lr_scheduler_True_weight_decay_0.1_seed_0"
            checkpoint_name = f"multimodal_text_encoder_lstm_embedding_dim_512_batch_size_8_dropout_i_0.5_lr_0.0001_lr_scheduler_True_weight_decay_0.1_max_epochs_400_seed_{args.seed}"
        if checkpoint_name.endswith(".ckpt"):
            checkpoint = checkpoint_name
        else:
            # grab checkpoint from epoch with lowest val loss
            checkpoint = glob.glob(
                f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]

        # load model from checkpoint
        model = MultiModalLitModel.load_from_checkpoint(
            checkpoint, map_location=device)
        model.eval()

        # parse empty args
        parser = _setup_parser()
        data_args = parser.parse_args("")

        # set args from checkpoint
        for key, value in model.args.items():
            setattr(data_args, key, value)

    # make the train dataloader deterministic
    data_args.augment_frames = False
    data_args.eval_include_sos_eos = args.eval_include_sos_eos
    data_args.eval_type = args.eval_type

    # get seed
    seed = args.seed

    # use clip for evaluation
    if args.model == "clip":
        data_args.clip_eval = True

    # set up object categories dataloader
    object_categories_dm = ObjectCategoriesDataModule(args)  # TODO: check this works
    object_categories_dm.prepare_data()
    object_categories_dm.setup()
    object_categories_dataloader = object_categories_dm.test_dataloader()

    # load vocab and metadata
    eval_data = load_data(EVAL_DATA_DIR / "eval_object_categories.json")
    vocab = _get_vocab()
    classes = _get_object_categories(vocab)

    # initialize correct and total pred counts
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # initialize results
    results = []

    # get model predictions
    for i, batch in enumerate(object_categories_dataloader):
        img, label, label_len, raw_label = batch

        # get text category label
        class_label = raw_label[0][0]

        if args.eval_type == "image":
            # perform evaluation using single category label with multiple images
            img = img.squeeze(0).to(device)  # remove outer batch
            label = label.to(device)
            label_len = label_len.to(device)

            # calculate similarity between images
            # first, get embeddings
            with torch.no_grad():
                if args.model == "clip":
                    label = label.squeeze(0)  # remove extra dim for CLIP
                    _, logits_per_text = model(img, label)
                else:
                    _, logits_per_text = model(img, label, label_len)

                logits_list = torch.softmax(logits_per_text,
                                            dim=-1).detach().cpu().numpy().tolist()[0]

                pred = torch.argmax(logits_per_text, dim=-1).item()
                ground_truth = 0
        elif args.eval_type == "text":
            # perform evaluation using single image with multiple category labels
            img = img.squeeze(0).to(device)
            label = label.squeeze(0).to(device)
            label_len = label_len.squeeze(0).to(device)

            # calculate similarity between images
            # first, get embeddings
            with torch.no_grad():
                if args.model == "clip":
                    label = label.squeeze(0)  # remove extra dim for CLIP
                    logits_per_image, _ = model(img, label)
                else:
                    logits_per_image, _ = model(img, label, label_len)

                logits_list = torch.softmax(logits_per_image, dim=-1).detach().cpu().numpy().tolist()[0]
                pred = torch.argmax(logits_per_image, dim=-1).item()
                ground_truth = 0
                
        # second, calculate if correct referent is predicted
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
            "checkpoint": checkpoint_name,
            "seed": seed,
            "trial_idx": i,
            "categories": curr_eval_categories,
            "logits": logits_list,
            "pred": pred,
            "correct": correct,
        }
        results.append(curr_results)

    # print accuracy for each class
    accuracies = []
    for classname, correct_count in correct_pred.items():
        accuracy = float(correct_count) / total_pred[classname]
        # print(f"Accuracy for class {classname:8s} is: {accuracy:.1%}")
        print(f"{classname}, {accuracy:.1%}")
        accuracies.append(accuracy)

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
        results_filename = f'results/{args.model}_seed_{seed}_{args.eval_type}_object_categories_predictions.json'
        print(results_filename)

        # save to JSON
        print(f"Saving predictions to {results_filename}")
        with open(results_filename, "w") as f:
            json.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm",
                        choices=['embedding', 'lstm', 'clip'],
                        help="which trained model to perform evaluations on")
    parser.add_argument("--seed", type=int, default=0,
                        help="which seed to load for trained model")
    parser.add_argument("--eval_include_sos_eos", action="store_true",
                        help="include SOS/EOS tokens for eval labels")
    parser.add_argument("--eval_type", type=str, default="image",
                        choices=["image", "text"],
                        help="Run evaluation using multiple images or multiple labels")
    parser.add_argument("--save_predictions", action="store_true",
                        help="save model predictions to JSON")
    args = parser.parse_args()

    main(args)
