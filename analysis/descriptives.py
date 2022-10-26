import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import cv2 as cv

DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
VIDEO_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/S_videos")
LABELED_S_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5")
OBJECT_CATEGORIES_ORIGINAL_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/object_categories_original")
OBJECT_CATEGORIES_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/object_categories")

pd.options.display.float_format = '{:.2f}'.format


def load_data():
    """Load multimodal saycam data"""
    with open(DATA_DIR / 'train.json') as f:
        train_data = json.load(f)
        train_data = train_data["data"]
        train_df = pd.DataFrame(train_data)
        train_df["split"] = "train"

    with open(DATA_DIR / 'val.json') as f:
        val_data = json.load(f)
        val_data = val_data["data"]
        val_df = pd.DataFrame(val_data)
        val_df["split"] = "val"

    with open(DATA_DIR / 'test.json') as f:
        test_data = json.load(f)
        test_data = test_data["data"]
        test_df = pd.DataFrame(test_data)
        test_df["split"] = "test"

    # concatenate train, val and test data
    multimodal_saycam_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return multimodal_saycam_df

def load_vocab():
    """Load saycam vocab"""
    with open(DATA_DIR / 'vocab.json') as f:
        vocab = json.load(f)
    return vocab

def calculate_dataset_descriptives(saycam_df, vocab):
    """Calculate descriptives for multimodal saycam dataset"""
    # order split column
    saycam_df["split"] = pd.Categorical(saycam_df["split"], categories=["train", "val", "test"], ordered=True)

    # get number of utterances per split
    utterances_per_split = saycam_df.groupby("split").size()
    print("Number of utterances in each split:")
    print(utterances_per_split)

    # get total number of utterances
    total_utterances = utterances_per_split.sum()
    print("Total number of utterances:")
    print(total_utterances)

    # get total number of tokens per split by summing the number of tokens in each utterance
    tokens_per_split = saycam_df.groupby("split")["utterance"].apply(lambda x: x.str.split().str.len().sum())
    print("Number of tokens in each split:")
    print(tokens_per_split)

    # get total number of tokens
    total_tokens = tokens_per_split.sum()
    print(f"Total number of tokens: {total_tokens}")

    # get mean and standard deviation of utterance length per split by splitting utterance

    # get sum of num_frames per split
    num_frames_per_split = saycam_df.groupby("split")["num_frames"].sum()
    print("Sum of num_frames in each split:")
    print(num_frames_per_split)

    # get total number of frames
    total_num_frames = num_frames_per_split.sum()
    print(f"Total number of frames: {total_num_frames}")

    # for each split, take each utterance and split, and calculate the length of each utterance
    saycam_df["utterance_length"] = saycam_df.groupby("split")["utterance"].apply(lambda x: x.str.split().str.len())

    # get mean utterance length per split
    mean_utterance_length_per_split = saycam_df.groupby("split")["utterance_length"].agg(["mean", "std"])
    print("Mean utterance length in each split:")
    print(mean_utterance_length_per_split)

    # get the average number of frames per utterance in each split
    mean_num_frames_per_utterance_per_split = num_frames_per_split / utterances_per_split
    print("Mean number of frames per utterance in each split:")
    print(mean_num_frames_per_utterance_per_split)

    # get size of vocab
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # plot utterance length distribution using seaborn
    utterance_length_distribution =  saycam_df["utterance_length"].value_counts().sort_index()

def calculate_date_descriptives(saycam_df):
    """Extract dates from transcripts"""
    saycam_df["date"] = saycam_df["frame_filenames"].apply(lambda x: x[0].split("_")[1])

    # convert to daytime with format YYYY-MM-DD
    saycam_df["date"] = pd.to_datetime(saycam_df["date"], format="%Y%m%d")

    # sort dates and get first and last entries (only in the training split)
    train_dates = saycam_df[saycam_df["split"] == "train"].sort_values("date")[["date", "split"]].drop_duplicates()
    print("Printing date descriptives")
    print(f"First transcribed date: {train_dates['date'].iloc[0]}")
    print(f"Last transcribed date: {train_dates['date'].iloc[-1]}")
    print(f"Number of months between first and last transcribed date: {(train_dates['date'].iloc[-1] - train_dates['date'].iloc[0]).days / 30} months and {(train_dates['date'].iloc[-1] - train_dates['date'].iloc[0]).days % 30} days")
    print(f"Number of unique transcribed dates: {len(train_dates)}")

    # get dates from original videos
    original_video_filenames = os.listdir(VIDEO_DIR)
    original_video_dates = pd.DataFrame([x.split("_")[1] for x in original_video_filenames], columns=["date"])
    original_video_dates = pd.to_datetime(original_video_dates["date"], format="%Y%m%d")
    # sort and print first and last date in original_video_dates
    original_video_dates = original_video_dates.sort_values()
    print(f"First original date: {original_video_dates.iloc[0]}")
    print(f"Last original date: {original_video_dates.iloc[-1]}")
    print(f"Number of unique original dates: {len(original_video_dates)}")

def calculate_video_descriptives(saycam_df):
    """Get video filenames to calculate total length of transcribed videos."""
    transcribed_video_filenames = saycam_df["video_filename"].unique()
    print(f"Number of transcribed videos: {len(transcribed_video_filenames)}")

    # get number of original videos
    original_video_filenames = os.listdir(VIDEO_DIR)
    print(f"Number of original videos: {len(original_video_filenames)}")

    # get proportion of transcribed videos
    print(f"Proportion of transcribed videos: {len(transcribed_video_filenames) / len(original_video_filenames)}")

    # get actual proportion by calculating minutes of transcribed videos vs. all videos
    transcribed_video_lengths = []
    for video_filename in transcribed_video_filenames:
        video_path = VIDEO_DIR / video_filename
        cap = cv.VideoCapture(str(video_path))
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count/(fps * 60))
        transcribed_video_lengths.append(duration)

    print(f"Total length of transcribed videos in minutes: {sum(transcribed_video_lengths)}")

    # get total length of all videos
    original_video_lengths = []
    for video_filename in original_video_filenames:
        video_path = VIDEO_DIR / video_filename
        cap = cv.VideoCapture(str(video_path))
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count/(fps * 60))
        original_video_lengths.append(duration)

    print(f"Total length of all videos in minutes: {sum(original_video_lengths)}")

    # get proportion of transcribed videos
    print(f"Proportion of transcribed videos: {sum(transcribed_video_lengths) / sum(original_video_lengths)}")

def calculate_labeled_s_descriptives(saycam_df):
    """Calculate descriptives for labeled S evaluation dataset."""
    # get number of images per category in Labeled S dataset
    labeled_s_categories = os.listdir(LABELED_S_DIR)
    labeled_s_categories.remove("carseat")
    labeled_s_categories.remove("couch")
    labeled_s_categories.remove("greenery")
    labeled_s_categories.remove("plushanimal")
    total_images = 0
    img_freq_dict = {}

    for category in labeled_s_categories:
        category_dir = LABELED_S_DIR / category
        num_images = len(os.listdir(category_dir))
        total_images += num_images
        print(f"Number of images in Labeled S {category}: {num_images}")
        img_freq_dict[category] = num_images
    print(f"Total number of images in Labeled S dataset: {total_images}")

    # get training split
    saycam_train_df = saycam_df[saycam_df["split"] == "train"]
    word_freq_dict = {}

    for category in labeled_s_categories:
        word_freq = 0
        for utterance in saycam_train_df['utterance'].tolist():
            words = utterance.split()
            if category in words:
                word_freq += 1
        word_freq_dict[category] = word_freq

    print("Word frequency of Labeled S categories in training split")
    print(word_freq_dict)

    # img_freq_norm = {k: v / sum(img_freq_dict.values()) for k, v in img_freq_dict.items()}
    # word_freq_norm = {k: v / sum(word_freq_dict.values()) for k, v in word_freq_dict.items()}
    img_freq_norm = {k: v for k, v in img_freq_dict.items()}
    word_freq_norm = {k: v for k, v in word_freq_dict.items()}

    # put into dataframe
    freq_df = pd.DataFrame.from_dict([img_freq_norm]).transpose().reset_index()
    freq_df.columns = ["category", "img_freq"]
    freq_df["word_freq"] = word_freq_norm.values()

    # create seaborn objects scatterplot
    # plot = (
    #     so.Plot(freq_df, x="img_freq", y="word_freq")
    #     .add(so.Dot())
    #     .add(so.Line(), so.PolyFit(order=1))
    #     .scale(x="log", y="log")
    # )

    # # save plot
    # plot.save("../figures/labeled-s-freq-scatterplot.png", dpi=600)

    print(freq_df)
    # create a matplotlib scatterplot of freq_df with img_freq on the x-axis and word_freq on the y-axis
    fig, ax = plt.subplots()
    ax.scatter(freq_df["img_freq"], freq_df["word_freq"])

    # add linear regression line
    # import scipy.stats as stats
    # slope, intercept, r_value, p_value, std_err = stats.linregress(freq_df["img_freq"], freq_df["word_freq"])
    # ax.plot(freq_df["img_freq"], intercept + slope * freq_df["img_freq"], 'r', label='fitted line')

    # add label for each point with category, img_freq, and word_freq
    for i, txt in enumerate(freq_df["category"]):
        ax.annotate(f"{txt} ({freq_df['img_freq'][i]}, {freq_df['word_freq'][i]})", (freq_df["img_freq"][i], freq_df["word_freq"][i]), size=6)
    
    # for i, txt in enumerate(freq_df["category"]):
    #     ax.annotate(txt, (freq_df["img_freq"].iloc[i], freq_df["word_freq"].iloc[i]), xytext=(0, 0), textcoords='offset points')
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Image frequency")
    ax.set_ylabel("Word frequency")
    plt.savefig("../figures/labeled-s-freq-scatterplot-mpl.png", dpi=600)
    

def calculate_object_categories_descriptives():
    """Calculate descriptives for object categories evaluation dataset."""
    # get number of categories/folders in original dataset
    original_categories = os.listdir(OBJECT_CATEGORIES_ORIGINAL_DIR)
    # filter by folders
    original_categories = [x for x in original_categories if os.path.isdir(OBJECT_CATEGORIES_ORIGINAL_DIR / x)]
    print(f"Number of object categories in original dataset: {len(original_categories)}")

    # get number of jpg images from folders under original dataset
    num_original_images = 0
    for category in original_categories:
        images = os.listdir(OBJECT_CATEGORIES_ORIGINAL_DIR / category)
        num_images = len([x for x in images if x.endswith(".jpg")])
        num_original_images += num_images
    print(f"Number of images in original dataset: {num_original_images}")
    print("Warning: this doesn't include test images")

    # get number of filtered object categories
    object_categories = os.listdir(OBJECT_CATEGORIES_DIR)
    # filter by folders
    object_categories = [x for x in object_categories if os.path.isdir(OBJECT_CATEGORIES_DIR / x)]
    print(f"Number of object categories in vocab: {len(object_categories)}")

    # get number of images from filtered object categories dataset
    num_images = 0
    for category in object_categories:
        images = os.listdir(OBJECT_CATEGORIES_DIR / category)
        num_images += len([x for x in images if x.endswith(".jpg")])
    print(f"Number of images in filtered object categories dataset: {num_images}")

    # load object categories JSON
    with open(DATA_DIR / "eval_object_categories.json", "r") as f:
        object_categories_eval = json.load(f)
        object_categories_eval = object_categories_eval["data"]

    # get length of object categories evaluation dataset
    print(f"Number of object categories evaluation trials: {len(object_categories_eval)}")

    # get labeled-s categories and get intersection with object categories
    labeled_s_categories = set(os.listdir(LABELED_S_DIR))
    object_categories = set(os.listdir(OBJECT_CATEGORIES_DIR))
    print(f"Number of object categories in both datasets: {len(labeled_s_categories.intersection(object_categories))}")
    print(f"Object categories in both datasets: {labeled_s_categories.intersection(object_categories)}")

def calculate_object_localization_descriptives():
    # TODO: only write code for this if we know we want to include this in the paper eventually!
    pass


def main():
    # load data and vocab
    saycam_df = load_data()
    vocab = load_vocab()

    # get descriptives
    calculate_dataset_descriptives(saycam_df, vocab)
    # calculate_date_descriptives(saycam_df)
    # calculate_video_descriptives(saycam_df)
    calculate_labeled_s_descriptives(saycam_df)
    # calculate_object_categories_descriptives()
    # calculate_object_localization_descriptives()

if __name__ == "__main__":
    main()
