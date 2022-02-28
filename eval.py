import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from multimodal.multimodal_data_module import EVAL_DATA_DIR, SOS_TOKEN_ID, EOS_TOKEN_ID
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.coco_captions_data_module import COCOCaptionsDataModule
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

    # build data module
    dataset_name = getattr(data_args, "dataset", "saycam")
    DataModuleClass = {
        "saycam": MultiModalSAYCamDataModule,
        "coco": COCOCaptionsDataModule,
    }[dataset_name]
    data = DataModuleClass(data_args)
    data.prepare_data()
    data.setup()

    vocab = data.read_vocab()

    # create dataloader
    eval_dataloader = {
        "dev": data.val_dataloader,
        "test": data.test_dataloader,
    }[args.dataset]()[1]

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
                        #choices=['embedding', 'lstm'],
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
