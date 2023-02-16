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
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.object_categories_data_module import ObjectCategoriesDataModule, _get_object_categories, _get_vocab
from multimodal.multimodal import MultiModalModel
from multimodal.multimodal_lit import MultiModalLitModel
from multimodal.attention_maps import gradCAM, getAttMap, n_inv, imshow
from train import _setup_parser

import clip

EVAL_FRAMES_DIRNAME = EVAL_DATA_DIR / "eval"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if args.clip_eval:
        print("Loading CLIP")
        checkpoint_name = "clip_vitl_14"
        model, preprocess = clip.load("ViT-L/14", device=device)
        model.eval()
        print("CLIP loaded")

        # set up parser
        parser = _setup_parser()
        data_args = parser.parse_args("")

        # set up CLIP config
        config = {}
        config["model"] = "clip"
        config["seed"] = None
        config["shuffle_utterances"] = None
        config["cnn"] = "clip"
        config["augment_frames"] = None
        config["multiple_frames"] = None
    else:
        checkpoint_name = args.checkpoint
        if args.checkpoint.endswith(".ckpt"):
            checkpoint = checkpoint_name
        elif "shuffle_utterances_True" in args.checkpoint:
            print('using last saved checkpoint')
            checkpoint = f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/last.ckpt"
        else:
            # grab checkpoint from epoch with lowest val loss
            checkpoint = glob.glob(
                f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/epoch*.ckpt")[0]
     
        # extract checkpoint config
        config = {}
        if "lstm" in checkpoint_name:
            config["model"] = "lstm"
        elif "embedding" in checkpoint_name:
            config["model"] = "embedding"
     
        if "seed_0" in checkpoint_name:
            config["seed"] = 0
        elif "seed_1" in checkpoint_name:
            config["seed"] = 1
        elif "seed_2" in checkpoint_name:
            config["seed"] = 2
        else:
            config["seed"] = None
     
        if "shuffle_utterances" in checkpoint_name:
            config["shuffle_utterances"] = True
        else:
            config["shuffle_utterances"] = False
     
        if "pretrained_cnn_True_finetune_cnn_True" in checkpoint_name:
            config["cnn"] = "finetune_pretrained"
        elif "pretrained_cnn_False_finetune_cnn_True" in checkpoint_name:
            config["cnn"] = "finetune_random_init"
        elif "pretrained_cnn_False_finetune_cnn_False" in checkpoint_name:
            config["cnn"] = "frozen_random_init"
        else:
            config["cnn"] = "frozen_pretrained"
     
        if "augment_frames_False" in checkpoint_name:
            config["augment_frames"] = False
        else:
            config["augment_frames"] = True
     
        if "multiple_frames_False" in checkpoint_name:
            config["multiple_frames"] = False
        else:
            config["multiple_frames"] = True

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
    # manually set to be filtered set
    data_args.eval_metadata_filename = args.eval_metadata_filename

    if args.clip_eval:
        # ensure CLIP transforms/tokenization are used
        data_args.clip_eval = True
    
    # build data module
    if args.eval_dataset == "saycam":
        # set up saycam dataloader
        stage = getattr(data_args, "stage", "saycam")
        data = MultiModalSAYCamDataModule(data_args)
        data.prepare_data()
        data.setup()

        # load vocab and metadata
        vocab = data.read_vocab()
        eval_data = load_data(EVAL_DATA_DIR / data_args.eval_metadata_filename)

        # create dataloader
        dataloader = {
            "dev": data.val_dataloader,
            "test": data.test_dataloader,
        }[args.stage]()[1]  # second dataloader contains eval data

        # get eval categories
        classes = sorted(os.listdir(EVAL_FRAMES_DIRNAME / "dev"))
    elif args.eval_dataset == "object_categories":
        # set up object categories dataloader
        object_categories_dm = ObjectCategoriesDataModule(
            data_args)  # TODO: check this works
        object_categories_dm.prepare_data()
        object_categories_dm.setup()
        dataloader = object_categories_dm.test_dataloader()

        # load vocab and metadata
        vocab = _get_vocab()
        eval_data = load_data(EVAL_DATA_DIR / "eval_object_categories.json")
        classes = _get_object_categories(vocab)

    # replace cat with kitty
    if args.use_kitty_label and not args.clip_eval:
        classes.remove("cat")
        classes.append("kitty")

    # initialize correct and total pred counts
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # initialize results
    results = []

    # get model predictions
    for i, batch in enumerate(dataloader):
        img, label, label_len, raw_label = batch

        # get text category label
        class_label = raw_label[0][0]

        if args.use_kitty_label and class_label == "cat" and not args.clip_eval:
            # use kitty for cat eval
            class_label = "kitty"

            if args.eval_type == "image":
                # replace single label
                label = [vocab[class_label]]
                if args.eval_include_sos_eos:
                    label = [SOS_TOKEN_ID] + label + [EOS_TOKEN_ID]
                label = torch.LongTensor([label])
            elif args.eval_type == "text":
                # replace true class label only
                label[0][0] = vocab[class_label]
                # ignoring SOS/EOS option for now...

        if args.eval_type == "image":
            # perform evaluation using single category label with multiple images
            img = img.squeeze(0).to(device)  # remove outer batch
            label = label.to(device)
            label_len = label_len.to(device)

            # calculate similarity between images
            # first, get embeddings
            with torch.no_grad():
                if args.clip_eval:
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
                if args.clip_eval:
                    label = label.squeeze(0)  # remove extra dim for CLIP
                    logits_per_image, _ = model(img, label)                    
                else:
                    logits_per_image, _ = model(img, label, label_len)
                logits_list = torch.softmax(logits_per_image,
                                            dim=-1).detach().cpu().numpy().tolist()[0]
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
            "model": config["model"],
            "seed": config["seed"],
            "shuffle_utterances": config["shuffle_utterances"],
            "augment_frames": config["augment_frames"],
            "multiple_frames": config["multiple_frames"],
            "cnn": config["cnn"],
            "eval_type": args.eval_type,
            "eval_dataset": args.eval_dataset,
            "stage": args.stage,
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
        if args.clip_eval:
            results_filename = f"results/{args.eval_dataset}/clip_{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
        elif config["shuffle_utterances"]:
            results_filename = f"results/{args.eval_dataset}/shuffle_{config['model']}_{config['cnn']}_seed_{config['seed']}_{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
        elif not config["augment_frames"]:
            results_filename = f"results/{args.eval_dataset}/{config['model']}_{config['cnn']}_augment_frames_{config['augment_frames']}_seed_{config['seed']}_{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
        elif not config["multiple_frames"]:
            results_filename = f"results/{args.eval_dataset}/{config['model']}_{config['cnn']}_multiple_frames_{config['multiple_frames']}_seed_{config['seed']}_{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"
        else:
            results_filename = f"results/{args.eval_dataset}/{config['model']}_{config['cnn']}_seed_{config['seed']}_{args.eval_type}_{args.eval_dataset}_{args.stage}_eval_predictions.json"

        # save to JSON
        print(f"Saving predictions to {results_filename}")
        with open(results_filename, "w") as f:
            json.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        help="path to checkpoint to use for evaluation")
    parser.add_argument("--clip_eval", action="store_true",
                       help="Use CLIP model for evaluation")
    parser.add_argument("--stage", type=str, default="test", choices=["dev", "test"],
                        help="which evaluation stage to use")
    parser.add_argument("--eval_include_sos_eos", action="store_true",
                        help="include SOS/EOS tokens for eval labels")
    parser.add_argument("--eval_type", type=str, default="image", choices=[
        "image", "text"], help="Run evaluation using multiple images or multiple labels")
    parser.add_argument("--eval_dataset", type=str, default="saycam", choices=[
                        "saycam", "object_categories"], help="Which evaludation dataset to use")
    parser.add_argument("--eval_metadata_filename", type=str,
                        default="eval_test.json",
                        help="JSON file with metadata evaluation split to use")
    parser.add_argument("--use_kitty_label", action="store_true",
                        help="replaces cat label with kitty")
    parser.add_argument("--save_predictions", action="store_true",
                        help="save model predictions to JSON")
    args = parser.parse_args()

    main(args)
