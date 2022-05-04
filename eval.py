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
from multimodal.multimodal import MultiModalModel
from multimodal.multimodal_lit import MultiModalLitModel
from multimodal.attention_maps import gradCAM, getAttMap, n_inv, imshow
from train import _setup_parser

EVAL_FRAMES_DIRNAME = EVAL_DATA_DIR / "eval"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # get checkpoint
    if args.model == "embedding":
        checkpoint_name = "multimodal_text_encoder_embedding_lr_0.0001_weight_decay_0.1_fix_temperature_True_batch_size_16"
    elif args.model == "lstm":
        checkpoint_name = 'multimodal_text_encoder_lstm_lr_0.0001_weight_decay_0.2_fix_temperature_False_batch_size_8'
    else:
        checkpoint_name = args.model
    if checkpoint_name.endswith(".ckpt"):
        checkpoint = checkpoint_name
    else:
        checkpoint = glob.glob(
            f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/*.ckpt")[0]

    # load model from checkpoint
    model = MultiModalLitModel.load_from_checkpoint(
        checkpoint, map_location=device)
    model.eval()

    # parse empty args
    parser = _setup_parser()
    data_args = parser.parse_args("")
    # set args
    for key, value in model.args.items():
        setattr(data_args, key, value)
    # make the train dataloader deterministic
    data_args.augment_frames = False
    data_args.eval_include_sos_eos = args.eval_include_sos_eos
    data_args.eval_type = args.eval_type
    data_args.eval_metadata_filename = args.eval_metadata_filename

    # build data module
    stage = getattr(data_args, "stage", "saycam")
    data = MultiModalSAYCamDataModule(data_args)
    data.prepare_data()
    data.setup()

    # load vocab and metadata
    vocab = data.read_vocab()
    eval_data = load_data(EVAL_DATA_DIR / args.eval_metadata_filename)

    # create dataloader
    eval_dataloader = {
        "dev": data.val_dataloader,
        "test": data.test_dataloader,
    }[args.stage]()[1]  # second dataloader contains eval data

    # get eval categories
    classes = sorted(os.listdir(EVAL_FRAMES_DIRNAME / "dev"))
    classes.remove("carseat")
    classes.remove("couch")
    classes.remove("greenery")
    classes.remove("plushanimal")

    # replace cat with kitty
    if args.use_kitty_label:
        classes.remove("cat")
        classes.append("kitty")

    # initialize correct and total pred counts
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # initialize results
    results = []

    # get model predictions
    for i, batch in enumerate(eval_dataloader):
        img, label, label_len, raw_label = batch

        # get text category label
        class_label = raw_label[0][0]

        if args.use_kitty_label and class_label == "cat":
            # use kitty for cat eval
            class_label = "kitty"
            label = [vocab[class_label]]
            if args.eval_include_sos_eos:
                label = [SOS_TOKEN_ID] + label + [EOS_TOKEN_ID]
            label = torch.LongTensor([label])

        if args.eval_type == "image":
            # perform evaluation using single category label with multiple images
            img = img.squeeze(0).to(device)  # remove outer batch
            label = label.to(device)
            label_len = label_len.to(device)

            # calculate similarity between images
            # first, get embeddings
            with torch.no_grad():
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
            "trial_idx": i,
            "categories": curr_eval_categories,
            "logits": logits_list,
            "pred": pred,
            "correct": correct,
        }
        results.append(curr_results)

        # plot attention map
        if args.plot_attention:
            # determine saliency layer to use
            saliency_layer = "layer4"

            # get text features
            text_features = model.model.encode_text(label, label_len)[0]
            if model.model.normalize_features:
                text_features = F.normalize(text_features, p=2, dim=1)

            # create attention map for current target image
            attn_map = gradCAM(
                model.vision_encoder.model,
                img[0].unsqueeze(0).to(device),
                text_features,
                getattr(model.vision_encoder.model, saliency_layer),
                normalize_features=model.model.normalize_features,
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()

            # get inverse image for plotting
            inv_img = img.squeeze(0)
            inv_img = n_inv(inv_img)
            np_img = inv_img[0].permute((1, 2, 0)).cpu().numpy()

            # save image
            os.makedirs('results', exist_ok=True)
            attention_map_filename = os.path.join(
                'results', f'{args.model}_{class_label}_{i % 100}_attn_map.png')
            fig, ax = plt.subplots()
            imshow(ax, getAttMap(np_img, attn_map))
            print(f'saving attention map: {attention_map_filename}')
            plt.savefig(attn_map_filename, bbox_inches='tight')
            plt.close()

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
        results_filename = f'results/{args.model}_{args.eval_type}_{args.eval_metadata_filename.replace(".json", "_predictions.json")}'
        print(results_filename)

        # save to JSON
        print(f"Saving predictions to {results_filename}")
        with open(results_filename, "w") as f:
            json.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm",
                        #choices=['embedding', 'lstm'],
                        help="which trained model to perform evaluations on")
    parser.add_argument("--stage", type=str, default="dev", choices=["dev", "test"],
                        help="which evaluation stage to use")
    parser.add_argument("--eval_include_sos_eos", action="store_true",
                        help="include SOS/EOS tokens for eval labels")
    parser.add_argument("--eval_type", type=str, default="image", choices=[
        "image", "text"], help="Run evaluation using multiple images or multiple labels")
    parser.add_argument("--eval_metadata_filename", type=str,
                        default="eval_dev.json",
                        help="JSON file with metadata for (dev) evaluation split to use")
    parser.add_argument("--use_kitty_label", action="store_true",
                        help="replaces cat label with kitty")
    parser.add_argument("--save_predictions", action="store_true",
                        help="save model predictions to CSV")
    parser.add_argument("--plot_attention", action="store_true",
                        help="plot attention maps for target images during eval")
    args = parser.parse_args()

    main(args)
