import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from multimodal.multimodal_data_module import read_vocab, LabeledSEvalDataset, multiModalDataset_collate_fn
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule, MultiModalSAYCamDataset
from multimodal.multimodal import MultiModalModel
from multimodal.multimodal_lit import MultiModalLitModel
from multimodal.attention_maps import gradCAM, viz_attn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # directories and filenames
    DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
    EVAL_FRAMES_DIRNAME = DATA_DIR / "eval"
    if args.dataset == "dev":
        EVAL_METADATA_FILENAME = DATA_DIR / "eval_dev.json"
    elif args.dataset == "test":
        EVAL_METADATA_FILENAME = DATA_DIR / "eval_test.json"

    VOCAB_FILENAME = DATA_DIR / "vocab.json"

    # load eval data
    with open(EVAL_METADATA_FILENAME) as f:
        eval_data = json.load(f)
        eval_data = eval_data["data"]

    vocab = read_vocab(VOCAB_FILENAME)  # read vocab
    vocab_idx2word = dict((v, k) for k, v in vocab.items())  # get mapping

    def label_to_category(i):
        """Returns category label for a given vocab index"""
        return vocab_idx2word[i.item()]

    # create dataloader
    eval_dataset = LabeledSEvalDataset(
        eval_data, vocab, eval_include_sos_eos=args.eval_include_sos_eos)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False)

    # get checkpoint
    if args.model == "embedding":
        checkpoint_name = "multimodal_text_encoder_embedding_lr_0.0001_weight_decay_0.1_fix_temperature_True_batch_size_16"
    elif args.model == "lstm":
        checkpoint_name = 'multimodal_text_encoder_lstm_lr_0.0001_weight_decay_0.2_fix_temperature_False_batch_size_8'

    # load model from checkpoint
    checkpoint = glob.glob(
        f"/home/wv9/code/WaiKeen/multimodal-baby/checkpoints/{checkpoint_name}/*.ckpt")[0]
    model = MultiModalLitModel.load_from_checkpoint(
        checkpoint, map_location=device)
    model.eval()

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
        img, label, label_len, _ = batch
        img = img.squeeze(0).to(device)  # remove outer batch
        label = label.to(device)
        label_len = label_len.to(device)

        # get text category label
        if args.eval_include_sos_eos:
            class_label = label_to_category(label[0][1])
        else:
            class_label = label_to_category(label)

        if args.use_kitty_label and class_label == "cat":
            # use kitty for cat eval
            if args.eval_include_sos_eos:
                label = torch.LongTensor(
                    [[vocab["<sos>"], vocab["kitty"], vocab["<eos>"]]]).to(device)
                class_label = "kitty"
            else:
                label = torch.LongTensor([[vocab["kitty"]]]).to(device)
                class_label = "kitty"

        # calculate similarity between images
        # first, get embeddings
        with torch.no_grad():
            _, logits_per_text = model(img, label, label_len)
            logits_list = torch.softmax(logits_per_text,
                                        dim=-1).detach().cpu().numpy().tolist()[0]
            pred = torch.argmax(logits_per_text, dim=-1).item()
            ground_truth = 0

        # second, calculate if correct referent is predicted
        correct = False
        if pred == ground_truth:
            correct = True
            correct_pred[class_label] += 1

        total_pred[class_label] += 1

        # store results
        curr_results = [checkpoint_name, i,
                        class_label, correct] + logits_list
        results.append(curr_results)

        # plot attention map
        if args.plot_attention:
            # determine saliency layer to use
            saliency_layer = "layer4"

            # create attention map for current target image
            attn_map = gradCAM(
                model.vision_encoder.model,
                img[0].unsqueeze(0).to(device),
                model.model.encode_text(label, label_len),
                getattr(model.vision_encoder.model, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()

            # get inverse image for plotting
            n_inv = transforms.Normalize(
                [-0.485/0.229, -0.546/0.224, -0.406/0.225],
                [1/0.229, 1/0.224, 1/0.225])
            inv_img = img.squeeze(0)
            inv_img = n_inv(inv_img)
            np_img = inv_img[0].permute((1, 2, 0)).cpu().numpy()

            # save image
            os.makedirs('results', exist_ok=True)
            attention_map_filename = os.path.join(
                'results', f'{args.model}_{class_label}_{i % 100}_attn_map.png')
            print(f'saving attention map: {attention_map_filename}')
            viz_attn(np_img, attn_map, attention_map_filename)

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))

    # print total accuracy
    total_correct = sum(correct_pred.values())
    total = sum(total_pred.values())
    print(f"Total accuracy: {total_correct / total}%")

    # save results
    if args.save_predictions:
        # create dir
        os.makedirs('results', exist_ok=True)

        # convert results to data frame
        columns = ['model_checkpoint', 'trial', 'category_label', 'correct',
                   'target_prob', 'foil_one_prob', 'foil_two_prob',
                   'foil_three_prob']
        results_df = pd.DataFrame(results, columns=columns)
        results_filename = os.path.join(
            "results", f"{args.model}_eval_predictions.csv")
        results_df.to_csv(results_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm",
                        choices=['embedding', 'lstm'],
                        help="which trained model to perform evaluations on")
    parser.add_argument("--dataset", type=str, default="dev", choices=["dev", "test"],
                        help="which evaluation dataset to use")
    parser.add_argument("--eval_include_sos_eos", action="store_true",
                        help="include SOS/EOS tokens for eval labels")
    parser.add_argument("--use_kitty_label", action="store_true",
                        help="replaces cat label with kitty")
    parser.add_argument("--save_predictions", action="store_true",
                        help="save model predictions to CSV")
    parser.add_argument("--plot_attention", action="store_true",
                        help="plot attention maps for target images during eval")
    args = parser.parse_args()

    main(args)
