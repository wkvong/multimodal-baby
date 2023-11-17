# compare frames from training and labeled-s eval for duplicates
# use both methods to detect exact and approximate duplicates

import cv2
import itertools
import json
import os
import pickle
import shutil
import subprocess
from tqdm import tqdm
from thefuzz import fuzz

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from multimodal.multimodal_saycam_data_module import DATA_DIR, EXTRACTED_FRAMES_DIRNAME, EVAL_FRAMES_DIRNAME, EVAL_TEST_METADATA_FILENAME
from multimodal.utils import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def ahash(image_path, hash_size=16):
    """Function to compute a perceptual hash"""

    # Load and convert image to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image
    resized = cv2.resize(image, (hash_size, hash_size))
    # Compute the mean pixel value
    mean_value = resized.mean()
    # Convert pixels to 1 or 0 based on comparison with the mean
    binary_matrix = (resized >= mean_value).astype(int)
    # Convert binary matrix to hash
    hash_value = sum([2 ** i for (i, v) in enumerate(binary_matrix.flatten()) if v])
    return hash_value


def calculate_exact_duplicates():
    """Calculate exact duplicates using a perceptual hash"""
    
    # get all eval frames
    # first, read in eval metadata
    with open(EVAL_TEST_METADATA_FILENAME) as f:
        eval_metadata = json.load(f)["data"]

    # second, get all eval frames (only target frames for now)
    eval_frames = []
    for i in range(len(eval_metadata)):
        eval_frames.append(eval_metadata[i]["target_img_filename"])
        # eval_frames.extend(eval_metadata[i]["foil_img_filenames"])

    print("number of eval frames:", len(eval_frames))

    # get all train frames
    TRAIN_FRAMES_DIR = DATA_DIR / EXTRACTED_FRAMES_DIRNAME

    train_frames = os.listdir(TRAIN_FRAMES_DIR)
    train_frames = [str(TRAIN_FRAMES_DIR / frame) for frame in train_frames]
    print("number of train frames:", len(train_frames))

    # check if results/duplicates/hashed before re-computing
    if os.path.exists("../results/duplicates/hashed_eval_frames.json") and os.path.exists("../results/duplicates/hashed_train_frames.json"):
        print("Loading hashed frames")

        # load hashed eval frames
        with open("../results/duplicates/hashed_eval_frames.json") as f:
            hashed_eval_frames = json.load(f)

        # load hashed train frames
        with open("../results/duplicates/hashed_train_frames.json") as f:
            hashed_train_frames = json.load(f)
    else:
        print("Calculating hashed frames")

        hashed_eval_frames = {}
        for image_path in tqdm(eval_frames):
            hashed_eval_frames[ahash(image_path)] = image_path

        hashed_train_frames = {}
        for image_path in tqdm(train_frames):
            hashed_train_frames[ahash(image_path)] = image_path

        # save hashed frames
        os.makedirs("../results/duplicates", exist_ok=True)
        with open("../results/duplicates/hashed_eval_frames.json", "w") as f:
            json.dump(hashed_eval_frames, f)
        with open("../results/duplicates/hashed_train_frames.json", "w") as f:
            json.dump(hashed_train_frames, f)

    # calculate duplicates and output file paths in either directory for matches, which are stored in the values of each dict
    duplicates = set(hashed_eval_frames.keys()).intersection(set(hashed_train_frames.keys()))

    # for duplicate in duplicates:
    #     # print duplicates separately on each line, with a new line afterwards
    #     print("eval frame:", hashed_eval_frames[duplicate])
    #     print("train frame:", hashed_train_frames[duplicate])
    #     print()

    print("total number of duplicates (using perceptual hash):", len(duplicates))    

    # get duplicates as a list
    duplicate_eval_frames = [hashed_eval_frames[duplicate] for duplicate in duplicates]
    duplicate_train_frames = [hashed_train_frames[duplicate] for duplicate in duplicates]

    # copy duplicate images to "../results/duplicates/img"
    os.makedirs("../results/duplicates/img", exist_ok=True)
    for img1, img2 in zip(duplicate_eval_frames, duplicate_train_frames):
        shutil.copy(img1, "../results/duplicates/img")
        shutil.copy(img2, "../results/duplicates/img")

    # create updated paths for the duplicate images pointing to the new location
    # by getting prepending "img" to the image filename
    updated_duplicate_eval_frames = ["img/" + os.path.basename(img) for img in duplicate_eval_frames]
    updated_duplicate_train_frames = ["img/" + os.path.basename(img) for img in duplicate_train_frames]

    # get utterances for duplicate training frames
    print("Getting utterances for duplicate training frames")
    TRAIN_METADATA = DATA_DIR / "train.json"
    VAL_METADATA = DATA_DIR / "val.json"
    TEST_METADATA = DATA_DIR / "test.json"

    with open(TRAIN_METADATA) as f:
        train_data = json.load(f)["data"]
    with open(VAL_METADATA) as f:
        val_data = json.load(f)["data"]
    with open(TEST_METADATA) as f:
        test_data = json.load(f)["data"]

    # combine train, val, and test data
    all_data = train_data + val_data + test_data    

    # get corresponding utterance from each duplicate train frame
    duplicate_utterances = []
    for frame in duplicate_train_frames:
        frame_basename = os.path.basename(frame)
        found = False
        for item in all_data:
            if frame_basename in item["frame_filenames"]:
                duplicate_utterances.append(item["utterance"])
                found = True

        if not found:
            print("corresponding utterance not found for", frame_basename)

    # get eval categories
    duplicate_eval_categories = [os.path.basename(os.path.dirname(img)) for img in duplicate_eval_frames]

    # create a html file which displays duplicates side-by-side with their filenames
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Image Duplicates</title>
      <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>

    <div class="container">
      <h2>Duplicate Images</h2>
      <div class="row">
        {image_blocks}
      </div>
    </div>

    </body>
    </html>
    """

    image_block_template = """
    <div class="col-md-2">
      <div class="card">
        <img class="card-img-top" src="{updated_image_path}" alt="Card image">
        <div class="card-body">
          <p class="card-text">{original_image_path}</p>
        </div>
      </div>
    </div>
    """

    image_blocks = ""

    for img1, img2, img1_old, img2_old in zip(updated_duplicate_eval_frames,
                                              updated_duplicate_train_frames,
                                              duplicate_eval_categories,
                                              duplicate_utterances):

        image_blocks += image_block_template.format(
            updated_image_path=img1,
            original_image_path=img1_old)
        image_blocks += image_block_template.format(
            updated_image_path=img2,
            original_image_path=img2_old)


    final_html = html_template.format(image_blocks=image_blocks)

    with open("../results/duplicates/duplicate_images.html", "w") as f:
        f.write(final_html)

    # calculate number of utterance and eval category matches
    duplicate_counts = 0
    for utterance, eval_category in zip(duplicate_utterances, duplicate_eval_categories):
        words = utterance.split(" ")
        if eval_category in words:
            print(f"eval category {eval_category} in utterance: {utterance}")
            duplicate_counts += 1

    print("number of utterance and eval category matches:", duplicate_counts)


def calculate_indirect_duplicates_with_hash():
    """Calculate indirect duplicates using perceptual hash"""
    
    # get all eval frames
    # first, read in eval metadata
    with open(EVAL_TEST_METADATA_FILENAME) as f:
        eval_metadata = json.load(f)["data"]

    # get train frames dir
    TRAIN_FRAMES_DIR = DATA_DIR / EXTRACTED_FRAMES_DIRNAME
    
    # get utterances for duplicate training frames
    print("Getting utterances for duplicate training frames")
    TRAIN_METADATA = DATA_DIR / "train.json"
    VAL_METADATA = DATA_DIR / "val.json"
    TEST_METADATA = DATA_DIR / "test.json"

    with open(TRAIN_METADATA) as f:
        train_data = json.load(f)["data"]
    with open(VAL_METADATA) as f:
        val_data = json.load(f)["data"]
    with open(TEST_METADATA) as f:
        test_data = json.load(f)["data"]

    # combine train, val, and test data
    all_data = train_data + val_data + test_data

    # get eval categories
    duplicate_eval_categories = sorted(
        os.listdir(DATA_DIR / EVAL_FRAMES_DIRNAME / "test"))

    # get all frame_filenames for each eval_category from training data
    eval_category_train_frame_filenames = {}
    for category in duplicate_eval_categories:
        eval_category_train_frame_filenames[category] = []
        for item in all_data:
            if category in item["utterance"].split(" "):
                eval_category_train_frame_filenames[category].extend(
                    item["frame_filenames"])

    # do the same thing for the eval data using eval_metadata
    eval_category_eval_frame_filenames = {}
    for category in duplicate_eval_categories:
        eval_category_eval_frame_filenames[category] = []  # init with empty list

    for item in eval_metadata:
        eval_category_eval_frame_filenames[item["target_category"]].append(
            item["target_img_filename"])
    
    # load ../results/duplicates/hashed_eval_frames_by_category.json if it exists
    # otherwise, compute perceptual hash for each eval frame and save to file
    if os.path.exists("../results/duplicates/hashed_eval_frames_by_category.json"):
        print("Loading perceptual hash data for evaluation frames")
        with open("../results/duplicates/hashed_eval_frames_by_category.json", "r") as f:
            hashed_eval_frames_by_category = json.load(f)
    else:
        # now compute perceptual hash for each frame in eval_category_eval_frame_filenames
        # replacing each item in the list with a tuple containing the (filename, hash)
        print("Calculating perceptual hash for evaluation frames")
        for category in duplicate_eval_categories:
            for i in tqdm(range(len(eval_category_eval_frame_filenames[category]))):
                frame_filename = eval_category_eval_frame_filenames[category][i]
                frame_path = DATA_DIR / EVAL_FRAMES_DIRNAME / "test" / category / frame_filename
                eval_category_eval_frame_filenames[category][i] = (
                    frame_filename, ahash(str(frame_path)))

        # rename as hashed_eval_frames_by_category
        hashed_eval_frames_by_category = eval_category_eval_frame_filenames

        # save this dict to a json file
        with open("../results/duplicates/hashed_eval_frames_by_category.json", "w") as f:
            json.dump(hashed_eval_frames_by_category, f)
        
    # do the same thing with eval_category_train_frame_filenames
    if os.path.exists("../results/duplicates/hashed_train_frames_by_category.json"):
        print("Loading perceptual hash data for training frames")
        with open("../results/duplicates/hashed_train_frames_by_category.json", "r") as f:
            hashed_train_frames_by_category = json.load(f)
    else:
        # now compute perceptual hash for each frame in eval_category_train_frame_filenames
        # replacing each item in the list with a tuple containing the (filename, hash)
        print("Calculating perceptual hash for training frames")
        for category in duplicate_eval_categories:
            for i in tqdm(range(len(eval_category_train_frame_filenames[category]))):
                frame_filename = eval_category_train_frame_filenames[category][i]
                frame_path = TRAIN_FRAMES_DIR / frame_filename
                eval_category_train_frame_filenames[category][i] = (
                    frame_filename, ahash(str(frame_path)))

        # rename as hashed_train_frames_by_category
        hashed_train_frames_by_category = eval_category_train_frame_filenames
                
        # save this dict to a json file
        with open("../results/duplicates/hashed_train_frames_by_category.json", "w") as f:
            json.dump(hashed_train_frames_by_category, f)

    # compare hashes between train and eval
    # if the hashes match, then the frames are duplicates
    # print out all duplicates from each category
    total_duplicates = 0
    for category in duplicate_eval_categories:
        # compute duplicates using python sets
        train_hashes = set([x[1] for x in hashed_train_frames_by_category[category]])
        eval_hashes = set([x[1] for x in hashed_eval_frames_by_category[category]])
        duplicates = train_hashes.intersection(eval_hashes)
        total_duplicates += len(duplicates)

    print(f"Total number of duplicates: {total_duplicates}")

    # now calculate indirect hashes
    # for each evaluation item, calculate the smallest distance to a training item from the same catgory
    hash_distances = []
    for category in duplicate_eval_categories:
        for eval_item in tqdm(hashed_eval_frames_by_category[category]):
            eval_frame, eval_hash = eval_item
            eval_hash = str(eval_hash)
            max_distance = 0
            max_train_frame = ""
            for train_item in hashed_train_frames_by_category[category]:
                train_frame, train_hash = train_item
                train_hash = str(train_hash)
                distance = fuzz.ratio(eval_hash, train_hash)

                if distance > max_distance:
                    max_distance = distance
                    max_train_frame = train_frame

            # save the eval frame, train frame, and distance
            hash_distances.append({
                "eval_frame": eval_frame,
                "train_frame": max_train_frame,
                "distance": max_distance
            })

    # save hash distances to JSON
    with open("../results/duplicates/indirect_duplicates.json", "w") as f:
        json.dump(hash_distances, f)

    # get all distances
    distances = [x["distance"] for x in hash_distances]

    # plot cumulative histogram
    import matplotlib.pyplot as plt
    plt.hist(distances, density=True, bins=100)

    # save image
    plt.savefig("../results/duplicates/indirect_duplicates_histogram.png")
    
def visualize_indirect_duplicates_with_hash():
    # load hash distances
    with open("../results/duplicates/indirect_duplicates.json", "r") as f:
        hash_distances = json.load(f)

    # sort hash_distances by decreasing distance
    hash_distances = sorted(hash_distances, key=lambda x: x["distance"], reverse=True)
    
    # create a html file which displays duplicates side-by-side with their filenames
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Image Duplicates</title>
      <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>

    <div class="container">
      <h2>Duplicate Images</h2>
      <div class="row">
        {image_blocks}
      </div>
    </div>

    </body>
    </html>
    """

    image_block_template = """
    <div class="col-md-2">
      <div class="card">
        <img class="card-img-top" src="{eval_image_path}" alt="Card image">
        <div class="card-body">
          <p class="card-text">{eval_image_path_str}</p>
        </div>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card">
        <img class="card-img-top" src="{train_image_path}" alt="Card image">
        <div class="card-body">
          <p class="card-text">{train_image_path_str}\nDistance: {distance}</p>
        </div>
      </div>
    </div>
    """

    image_blocks = ""

    # plot all items from hash_distances (train_frame, eval_frame, distance)
    # replace actual paths with the ../results/duplicates/img + filename
    for item in hash_distances:
        image_blocks += image_block_template.format(
            eval_image_path="img/" + os.path.basename(item["eval_frame"]),
            eval_image_path_str=item["eval_frame"],
            train_image_path="img/" + os.path.basename(item["train_frame"]),
            train_image_path_str=item["train_frame"],
            distance=item["distance"]
        )

    final_html = html_template.format(image_blocks=image_blocks)

    # copy all images to ../results/duplicates/img
    # for item in tqdm(hash_distances):
    #     eval_frame = item["eval_frame"]
    #     train_frame = item["train_frame"]
    #     shutil.copy(eval_frame, "../results/duplicates/img")
    #     shutil.copy(str(DATA_DIR / EXTRACTED_FRAMES_DIRNAME / train_frame), "../results/duplicates/img")

    with open("../results/duplicates/indirect_duplicate_images.html", "w") as f:
        f.write(final_html)
    
 
def calculate_indirect_duplicates_with_nn_features():
    """Calculate indirect duplicates using NN features"""

    # get all eval frames
    # first, read in eval metadata
    with open(EVAL_TEST_METADATA_FILENAME) as f:
        eval_metadata = json.load(f)["data"]

    # get train frames dir
    TRAIN_FRAMES_DIR = DATA_DIR / EXTRACTED_FRAMES_DIRNAME
    
    # get utterances for duplicate training frames
    print("Getting utterances for duplicate training frames")
    TRAIN_METADATA = DATA_DIR / "train.json"
    VAL_METADATA = DATA_DIR / "val.json"
    TEST_METADATA = DATA_DIR / "test.json"

    with open(TRAIN_METADATA) as f:
        train_data = json.load(f)["data"]
    with open(VAL_METADATA) as f:
        val_data = json.load(f)["data"]
    with open(TEST_METADATA) as f:
        test_data = json.load(f)["data"]

    # combine train, val, and test data
    all_data = train_data + val_data + test_data

    # get eval categories
    duplicate_eval_categories = sorted(
        os.listdir(DATA_DIR / EVAL_FRAMES_DIRNAME / "test"))

    # get all frame_filenames for each eval_category from training data
    eval_category_train_frame_filenames = {}
    for category in duplicate_eval_categories:
        eval_category_train_frame_filenames[category] = []
        for item in all_data:
            if category in item["utterance"].split(" "):
                eval_category_train_frame_filenames[category].extend(
                    item["frame_filenames"])

    # do the same thing for the eval data using eval_metadata
    eval_category_eval_frame_filenames = {}
    for category in duplicate_eval_categories:
        eval_category_eval_frame_filenames[category] = []  # init with empty list
    
    for item in eval_metadata:
        eval_category_eval_frame_filenames[item["target_category"]].append(
            item["target_img_filename"])

    # load model
    model_name = "dino_sfp_resnext50"
    model = load_model(model_name, pretrained=True)
    model = model.eval()
    model = model.to(device)
    
    # embed train images (eval_category_train_frame_filenames) and save features
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # check if train features have been computed
    if os.path.exists("../results/duplicates/train_features.pkl"):
        with open("../results/duplicates/train_features.pkl", "rb") as f:
            train_features = pickle.load(f)
    else:
        train_features = {}
        for category in duplicate_eval_categories:
            print("Embedding train images for", category)
            # get list of frames
            frames = [TRAIN_FRAMES_DIR / img for img in eval_category_train_frame_filenames[category]]
            curr_features = []

            # embed frames
            for frame in tqdm(frames):
                # load image
                img = transform(Image.open(frame))
                img = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model(img)
                curr_features.append(features.detach().cpu().numpy())

            train_features[category] = np.concatenate(curr_features, axis=0)

        # save train features
        with open("../results/duplicates/train_features.pkl", "wb") as f:
            pickle.dump(train_features, f)

    # do the same thing with eval frames
    # check if eval features have been computed
    if os.path.exists("../results/duplicates/eval_features.pkl"):
        with open("../results/duplicates/eval_features.pkl", "rb") as f:
            eval_features = pickle.load(f)
    else:
        eval_features = {}
        for category in duplicate_eval_categories:
            print("Embedding eval images for", category)
            # get list of frames
            frames = eval_category_eval_frame_filenames[category]
            curr_features = []

            # embed frames
            for frame in tqdm(frames):
                # load image
                img = transform(Image.open(frame))
                img = img.unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model(img)
                curr_features.append(features.detach().cpu().numpy())

            eval_features[category] = np.concatenate(curr_features, axis=0)

        # save eval features
        with open("../results/duplicates/eval_features.pkl", "wb") as f:
            pickle.dump(eval_features, f)
        
    # calculate cosine similarity between train and eval features
    # check if cosine similarity has been computed
    if os.path.exists("../results/duplicates/cosine_similarity.pkl"):
        with open("../results/duplicates/cosine_similarity.pkl", "rb") as f:
            cosine_similarity = pickle.load(f)
    else:
        cosine_similarity = {}
        for category in duplicate_eval_categories:
            print("Calculating cosine similarity for", category)
            # convert to tensors
            train_features_tensor = torch.from_numpy(train_features[category]).to(device)
            eval_features_tensor = torch.from_numpy(eval_features[category]).to(device)

            # normalize features
            train_features_tensor = F.normalize(train_features_tensor, dim=-1)
            eval_features_tensor = F.normalize(eval_features_tensor, dim=-1)

            # add fake dimensions
            train_features_tensor = train_features_tensor[:, None, :]
            eval_features_tensor = eval_features_tensor[None, :, :]

            # calculate cosine sim
            curr_cosine_similarity = F.cosine_similarity(
                train_features_tensor, eval_features_tensor, dim=-1).detach().cpu().numpy()

            # get train and eval frame filenames
            train_frame_filenames = eval_category_train_frame_filenames[category]
            eval_frame_filenames = eval_category_eval_frame_filenames[category]

            # put together as dict, and add to cosine_similarity dict
            cosine_similarity[category] = {
                'train_frames': train_frame_filenames,
                'eval_frames': eval_frame_filenames,
                'cosine_similarity': curr_cosine_similarity
            }

        # save cosine similarity
        with open("../results/duplicates/cosine_similarity.pkl", "wb") as f:
            pickle.dump(cosine_similarity, f)
            
    # get top match for each eval frame
    max_cosine_sims = []
    for category in duplicate_eval_categories:
        print("Getting top match for", category)
        # get cosine similarity
        curr_cosine_similarity = cosine_similarity[category]["cosine_similarity"]

        # loop over each column
        for i in range(curr_cosine_similarity.shape[1]):
            # get max cosine similarity
            max_cosine_sim_idx = np.argmax(curr_cosine_similarity[:, i])
            max_cosine_sim = np.max(curr_cosine_similarity[:, i])
            eval_frame = cosine_similarity[category]["eval_frames"][i]
            train_frame = cosine_similarity[category]["train_frames"][max_cosine_sim_idx]
            max_cosine_sims.append({
                "train_frame": train_frame,
                "eval_frame": eval_frame,
                "max_cosine_sim": max_cosine_sim
            })

    # copy train frames to ../results/duplicates/img
    # for item in tqdm(max_cosine_sims):
    #     shutil.copy(str(DATA_DIR / EXTRACTED_FRAMES_DIRNAME / item["train_frame"]), "../results/duplicates/img")
            
    # save max cosine sims as pickle
    with open("../results/duplicates/max_cosine_sims.pkl", "wb") as f:
        pickle.dump(max_cosine_sims, f)

def visualize_indirect_duplicates_with_nn_features():        
    # load max cosine sims
    with open("../results/duplicates/max_cosine_sims.pkl", "rb") as f:
        max_cosine_sims = pickle.load(f)

    # sort cosine sims by decreasing distance
    max_cosine_sims = sorted(max_cosine_sims, key=lambda x: x["max_cosine_sim"], reverse=True)
    
    # create a html file which displays duplicates side-by-side with their filenames
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Image Duplicates</title>
      <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>

    <div class="container">
      <h2>Duplicate Images</h2>
      <div class="row">
        {image_blocks}
      </div>
    </div>

    </body>
    </html>
    """

    image_block_template = """
    <div class="col-md-2">
      <div class="card">
        <img class="card-img-top" src="{eval_image_path}" alt="Card image">
        <div class="card-body">
          <p class="card-text">{eval_image_path_str}</p>
        </div>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card">
        <img class="card-img-top" src="{train_image_path}" alt="Card image">
        <div class="card-body">
          <p class="card-text">{train_image_path_str}\nCosine Sim: {max_cosine_sim}</p>
        </div>
      </div>
    </div>
    """

    image_blocks = ""

    # plot all items from max cosine sims (train_frame, eval_frame, distance)
    # replace actual paths with the ../results/duplicates/img + filename
    for item in max_cosine_sims:
        image_blocks += image_block_template.format(
            eval_image_path="img/" + os.path.basename(item["eval_frame"]),
            eval_image_path_str=item["eval_frame"],
            train_image_path="img/" + os.path.basename(item["train_frame"]),
            train_image_path_str=item["train_frame"],
            max_cosine_sim=item["max_cosine_sim"]
        )

    final_html = html_template.format(image_blocks=image_blocks)

    with open("../results/duplicates/indirect_duplicate_images_with_nn_features.html", "w") as f:
        f.write(final_html)

def plot_cosine_sims():
    # load max cosine sims
    with open("../results/duplicates/max_cosine_sims.pkl", "rb") as f:
        max_cosine_sims = pickle.load(f)

    # extract max cosine sims as a list
    max_cosine_sims = [item["max_cosine_sim"] for item in max_cosine_sims]

    # plot as histogram with 20 bins and add some spacing between bins
    # convert y-axis to proportion
    # set bins to be 0 to 1, with 0.05 in between
    plt.hist(max_cosine_sims, bins=np.arange(0, 1.05, 0.05), rwidth=0.9)
    plt.xlim(0, 1)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Max Cosine Similarity between Training and\nEvaluation frames from the same category")
    plt.savefig("../results/duplicates/cosine_similarity.png", dpi=300)

    # calculate proportion between 0.7 and 0.8
    print("Proportion of max cosine sims between 0.7 and 0.8:", np.sum(np.logical_and(np.array(max_cosine_sims) >= 0.7, np.array(max_cosine_sims) < 0.8)) / len(max_cosine_sims))

    # calculate proportion between 0.8 and 0.9
    print("Proportion of max cosine sims between 0.8 and 0.9:", np.sum(np.logical_and(np.array(max_cosine_sims) >= 0.8, np.array(max_cosine_sims) < 0.9)) / len(max_cosine_sims))
    
    # calculate proportion between 0.9 and 1
    print("Proportion of max cosine sims between 0.9 and 1:", np.sum(np.array(max_cosine_sims) >= 0.9) / len(max_cosine_sims))

def calculate_nearest_neighbors():
    print("Calculating nearest neighbors...")
    
    # get eval features
    with open("../results/duplicates/eval_features.pkl", "rb") as f:
        eval_features = pickle.load(f)

    # get train features
    with open("../results/duplicates/train_features.pkl", "rb") as f:
        train_features = pickle.load(f)

    # get utterances for duplicate training frames
    # first, read in eval metadata
    with open(EVAL_TEST_METADATA_FILENAME) as f:
        eval_metadata = json.load(f)["data"]
        
    print("Getting utterances for duplicate training frames")
    TRAIN_METADATA = DATA_DIR / "train.json"
    VAL_METADATA = DATA_DIR / "val.json"
    TEST_METADATA = DATA_DIR / "test.json"

    with open(TRAIN_METADATA) as f:
        train_data = json.load(f)["data"]
    with open(VAL_METADATA) as f:
        val_data = json.load(f)["data"]
    with open(TEST_METADATA) as f:
        test_data = json.load(f)["data"]

    # combine train, val, and test data
    all_data = train_data + val_data + test_data    
        
    # get eval and train filenames
    duplicate_eval_categories = sorted(
        os.listdir(DATA_DIR / EVAL_FRAMES_DIRNAME / "test"))

    # get all frame_filenames for each eval_category from training data
    train_filenames = {}
    for category in duplicate_eval_categories:
        train_filenames[category] = []
        for item in all_data:
            if category in item["utterance"].split(" "):
                train_filenames[category].extend(
                    item["frame_filenames"])

    # combine all values from train_filenames into a single list
    train_filenames = list(itertools.chain(*train_filenames.values()))

    # do the same thing for the eval data using eval_metadata
    eval_filenames = {}
    for category in duplicate_eval_categories:
        eval_filenames[category] = []  # init with empty list
    
    for item in eval_metadata:
        eval_filenames[item["target_category"]].append(
            item["target_img_filename"])
        
    # get train labels
    train_labels = np.concatenate([[label] * len(train_features[label]) for label in train_features.keys()])
        
    # concatenate all train features, and convert to torch tensor
    train_features = np.concatenate(list(train_features.values()), axis=0)
    train_features = torch.from_numpy(train_features).to(device)
    train_features = train_features[None, :, :]

    # get eval categories
    duplicate_eval_categories = sorted(
        os.listdir(DATA_DIR / EVAL_FRAMES_DIRNAME / "test"))

    total_accuracy = 0

    matched_cosine_sims = []
    mismatched_cosine_sims = []

    # save filenames
    matched_train_filenames = []
    matched_eval_filenames = []

    # save results
    matched_results = []
    
    for i, eval_category in tqdm(enumerate(duplicate_eval_categories)):
        accuracy = 0
        category_eval_features = eval_features[eval_category]
        # convert to torch tensor
        category_eval_features = torch.from_numpy(category_eval_features).to(device)

        # add fake dims to eval features
        category_eval_features = category_eval_features[:, None, :]
        
        # compute cosine sims to all train features
        cosine_sims = F.cosine_similarity(category_eval_features, train_features, dim=-1)

        # get max cosine details
        max_cosine_sims = torch.max(cosine_sims, dim=-1).values.detach().cpu().numpy()
        max_cosine_indices = torch.argmax(cosine_sims, dim=-1).detach().cpu().numpy()
        max_cosine_labels = [train_labels[idx] for idx in max_cosine_indices]
        max_cosine_eval_filenames = [eval_filenames[eval_category][idx] for idx in range(len(max_cosine_indices))]
        max_cosine_train_filenames = [train_filenames[idx] for idx in max_cosine_indices]

        for j, label in enumerate(max_cosine_labels):
            if label == eval_category:
                accuracy += 1
                matched_cosine_sims.append(max_cosine_sims[j])
                matched_train_filenames.append(
                    max_cosine_train_filenames[j])
                matched_eval_filenames.append(
                    max_cosine_eval_filenames[j])
                matched_results.append([
                    max_cosine_eval_filenames[j],
                    max_cosine_train_filenames[j],
                    max_cosine_sims[j],
                    "match"])
            else:
                mismatched_cosine_sims.append(max_cosine_sims[j])
                matched_results.append([
                    max_cosine_eval_filenames[j],
                    max_cosine_train_filenames[j],
                    max_cosine_sims[j],
                    "mismatch"])

        print(f"Accuracy for {eval_category}: {accuracy / 100}")
        total_accuracy += accuracy

    print(f"Total accuracy: {total_accuracy / 2200}")

    print("Total of matched cosine sims > 0.999:", np.sum(np.array(matched_cosine_sims) > 0.99)) 
    
    # get the proportion of matched cosine sims > 0.95
    print("Proportion of matched cosine sims > 0.95:", np.sum(np.array(matched_cosine_sims) > 0.95) / (len(matched_cosine_sims) + len(mismatched_cosine_sims)))

    # do the same thing but with a 0.9 threshold
    print("Proportion of matched cosine sims > 0.9:", np.sum(np.array(matched_cosine_sims) > 0.9) / (len(matched_cosine_sims) + len(mismatched_cosine_sims)))
    
    # create histogram plot
    plt.hist([mismatched_cosine_sims, matched_cosine_sims], bins=np.arange(0, 1.05, 0.05), rwidth=0.9, stacked=False)
    plt.xlim(0, 1)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    # plt.title("Cosine Similarity between Training and\nEvaluation frames")
    plt.legend(["Mismatched", "Matched"])
    plt.savefig(
"../results/duplicates/stacked_histogram_nearest_neighbors_cosine_similarity.png", dpi=300)

    # save train and eval filenames with matched cosine sims as pickle
    with open("../results/duplicates/matched_filenames.pkl", "wb") as f:
        pickle.dump({"train": matched_train_filenames,
                     "eval": matched_eval_filenames,
                     "sims": matched_cosine_sims}, f)

    # save results as csv using pandas
    with open("../results/duplicates/matched_results.csv", "w") as f:
        # columns: ["eval_filename", "train_filename", "cosine_sim", "matched"]
        results = pd.DataFrame(matched_results, columns=["eval_filename", "train_filename", "cosine_sim", "matched"])
        results.to_csv(f, index=False)
        

def analyze_nearest_neighbors():
    with open("../results/duplicates/matched_filenames.pkl", "rb") as f:
        matched_filenames = pickle.load(f)
        matched_train_filenames = matched_filenames["train"]
        matched_eval_filenames = matched_filenames["eval"]
        matched_cosine_sims = matched_filenames["sims"]

    # get index for sim closest to 0.95 in value
    sims = [0.99, 0.94, 0.9, 0.86, 0.8, 0.75]
    for sim in sims:
        closest_idx = np.argmin(np.abs(np.array(matched_cosine_sims) - sim))
        closest_sim = np.array(matched_cosine_sims)[closest_idx]
        print(f"sim: {closest_sim}")
        print(f"closest train filename: {matched_train_filenames[closest_idx]}")
        # use subprocess and imgcat to visualize the image
        subprocess.run(["imgcat", str(DATA_DIR / EXTRACTED_FRAMES_DIRNAME / matched_train_filenames[closest_idx])])
        print(f"closest eval filename: {matched_eval_filenames[closest_idx]}")
        subprocess.run(["imgcat", str(matched_eval_filenames[closest_idx])])
        
    
        
# create custom dataset for train images
class TrainFramesDataset(Dataset):
    def __init__(self, train_frames, train_labels):
        self.train_frames = train_frames
        self.train_labels = train_labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.train_frames)

    def __getitem__(self, idx):
        # load image
        image = Image.open(DATA_DIR / EXTRACTED_FRAMES_DIRNAME / self.train_frames[idx]).convert("RGB")
        image = self.transform(image)
        label = self.train_labels[idx]
        frame = self.train_frames[idx]

        return image, label, frame


def calculate_nearest_neighbors_in_pixel_space():
    """Same as above but in pixel space instead"""
    
    # get eval categories
    duplicate_eval_categories = sorted(
        os.listdir(DATA_DIR / EVAL_FRAMES_DIRNAME / "test"))
    
    # get eval images
    # first, read in eval metadata
    with open(EVAL_TEST_METADATA_FILENAME) as f:
        eval_metadata = json.load(f)["data"]

    # second, get all eval frames (only target frames for now)
    eval_frames = []
    eval_labels = []
    for i in range(len(eval_metadata)):
        eval_frames.append(eval_metadata[i]["target_img_filename"])
        eval_labels.append(eval_metadata[i]["target_category"])

    # print length and check they are the same
    print("Number of eval frames:", len(eval_frames))
    print("Number of eval labels:", len(eval_labels))
    assert len(eval_frames) == len(eval_labels)

    # get train frames dir
    TRAIN_FRAMES_DIR = DATA_DIR / EXTRACTED_FRAMES_DIRNAME
    
    # get utterances for duplicate training frames
    print("Getting utterances for duplicate training frames")
    TRAIN_METADATA = DATA_DIR / "train.json"
    VAL_METADATA = DATA_DIR / "val.json"
    TEST_METADATA = DATA_DIR / "test.json"

    with open(TRAIN_METADATA) as f:
        train_data = json.load(f)["data"]
    with open(VAL_METADATA) as f:
        val_data = json.load(f)["data"]
    with open(TEST_METADATA) as f:
        test_data = json.load(f)["data"]

    # combine train, val, and test data
    all_data = train_data + val_data + test_data

    # get all frame_filenames for each eval_category from training data
    train_frames = []
    train_labels = []
    for category in duplicate_eval_categories:
        for item in all_data:
            if category in item["utterance"].split(" "):
                train_frames.append(item["frame_filenames"][0])
                train_labels.append(category)

    assert len(train_frames) == len(train_labels)                                    

    # create dataset/dataloader
    train_dataset = TrainFramesDataset(train_frames, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False)

    # transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    accuracy = 0
    results = []

    for eval_frame, eval_label in tqdm(zip(eval_frames, eval_labels)):
        # load image
        eval_img = transform(Image.open(eval_frame).convert("RGB")).to(device)

        min_distance = float("inf")
        min_label = ""
        min_frame = ""

        for train_images, train_labels, train_frames in train_dataloader:
            # move images to gpu
            train_images = train_images.to(device)
            
            # compute raw pixel distance
            distance = torch.sum(torch.abs(eval_img - train_images), dim=(1, 2, 3))

            # get min distance and label
            curr_min_distance = torch.min(distance)
            curr_min_label = train_labels[torch.argmin(distance)]
            curr_min_frame = train_frames[torch.argmin(distance)]
            if curr_min_distance < min_distance:
                min_distance = curr_min_distance
                min_label = curr_min_label
                min_frame = curr_min_frame

        # print eval and min label
        print()
        print(f"Eval label: {eval_label}, Min Train Label: {min_label}")

        # compare label to eval label
        if min_label == eval_label:
            accuracy += 1

        # append to results, and detach any variables if necessary
        results.append({
            "eval_frame": eval_frame,
            "eval_label": eval_label,
            "min_label": min_label,
            "min_frame": min_frame,
            "min_distance": min_distance,
            "correct": min_label == eval_label
        })
            
    print(f"Accuracy: {accuracy / 2200}")

    # save results as JSON
    with open("../results/duplicates/nn_pixel_space_results.json", "w") as f:
        json.dump(results, f)

     
if __name__ == "__main__":
    # calculate_direct_duplicates()
    # calculate_indirect_duplicates_with_hash()
    # visualize_indirect_duplicates_with_hash()
    # calculate_indirect_duplicates_with_nn_features()
    # visualize_indirect_duplicates_with_nn_features()
    # plot_cosine_sims()
    calculate_nearest_neighbors()
    analyze_nearest_neighbors()
    # calculate_nearest_neighbors_in_pixel_space()
