import glob
import json
import numpy as np
import pandas as pd
from siuba import group_by, summarize, filter, mutate, if_else, _
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

# plot settings
matplotlib.use('module://imgcat')
sns.set(rc={"figure.dpi":300, "savefig.dpi":300})

def collate_saycam_results():
    saycam_results_filenames = glob.glob("../results/*saycam*.json")
     
    saycam_results = []
    for filename in saycam_results_filenames:
        with open(filename) as f:
            data = json.load(f)
     
        result_df = pd.DataFrame(data["data"])
        result_df["target_category"] = result_df["categories"].str[0]
        saycam_results.append(result_df)
     
    # combine results
    saycam_results_df = pd.concat(saycam_results)
    return saycam_results_df

def collate_brady_results():
    # get object categories results
    brady_results_filenames = glob.glob("../results/*object_categories*.json")
     
    brady_results = []
    for filename in brady_results_filenames:
        with open(filename) as f:
            data = json.load(f)
     
        result_df = pd.DataFrame(data["data"])
        result_df["target_category"] = result_df["categories"].str[0]
        brady_results.append(result_df)
     
    # combine results
    brady_results_df = pd.concat(brady_results)

def get_saycam_summary_diff_init(saycam_results_df):
    """Get SAYCam evaluation accuracy for different CNN training conditions"""
    saycam_results_summary_diff_init = (
    saycam_results_df 
        >> filter(_.shuffle_utterances == False)
        >> group_by(_.cnn, _.model, _.seed, _.eval_type)
        >> summarize(correct = _.correct.mean()
    ))
    return saycam_results_summary_diff_init
    
def plot_saycam_summary_diff_init():
    """Grouped bar plot of SAYCam accuracy for different CNN training conditions"""
    saycam_results_df = collate_saycam_results()
    saycam_results_summary_diff_init = get_saycam_summary_diff_init(saycam_results_df)
    sns.catplot(x="model", y="correct", hue="cnn", kind="bar",
                col="eval_type", errorbar=('ci', 68), data=saycam_results_summary_diff_init)
    plt.show()
    plt.savefig("../figures/labeled-s-summary-diff-init.png", dpi=600)

def get_brady_summary_diff_init():
    """Get Brady evaluation accuracy for different CNN training conditions"""
    brady_results_df = collate_brady_results()
    brady_results_summary_diff_init = (brady_results_df
    >> filter(_.shuffle_utterances == False)
    >> group_by(_.cnn, _.model, _.seed, _.eval_type)
    >> summarize(correct = _.correct.mean()))
    return brady_results_summary_diff_init

def plot_brady_summary_diff_init():
    """Grouped bar plot of Brady accuracy for different CNN training conditions"""
    brady_results_summary_diff_init = get_brady_summary_diff_init()
    sns.catplot(x="model", y="correct", hue="cnn", kind="bar",
                col="eval_type", ci=68, data=brady_results_summary_diff_init)
    plt.show()

def get_saycam_summary_by_target_category():
    """Get SAYCam evaluation accuracy for different target categories"""
    saycam_results_df = collate_saycam_results()
    saycam_target_category_summary_df = (
    saycam_results_df
        >> filter(_.shuffle_utterances == False, _.cnn == "frozen_pretrained")
        >> group_by(_.model, _.seed, _.eval_type, _.target_category)
        >> summarize(correct = _.correct.mean()))
    return saycam_target_category_summary_df

def plot_saycam_target_category_summary():
    """Grouped bar plot of SAYCam accuracy for different target categories"""
    saycam_target_category_summary_df = get_saycam_summary_by_target_category()
    plt.figure(figsize=(20, 15))
    cat = sns.catplot(x="target_category", y="correct", hue="model", kind="bar",
                row="eval_type", ci=68, data=saycam_target_category_summary_df, legend=False)
    for ax in cat.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.show()

def get_saycam_summary_shuffled():
    """Get SAYCam evaluation accuracy for matched vs. shuffled utterances"""
    saycam_results_df = collate_saycam_results()
    saycam_results_shuffled_summary = (
    saycam_results_df
      >> filter(_.cnn == "frozen_pretrained")
      >> group_by(_.model, _.seed, _.eval_type, _.shuffle_utterances)
      >> summarize(correct = _.correct.mean())
    )
    return saycam_results_shuffled_summary

# TODO: plot saycam shuffled results

def get_brady_summary_by_target_category(brady_results_df):
    """Get Brady evaluation accuracy for different target categories"""
    brady_results_summary_by_target_category = (
    brady_results_df
    >> filter(_.shuffle_utterances == False, _.cnn == "frozen_pretrained")
    >> group_by(_.model, _.seed, _.eval_type, _.target_category)
    >> summarize(correct = _.correct.mean()))
    return brady_results_summary_by_target_category

def plot_brady_target_category_summary(brady_results_summary_by_target_category):
    """Grouped bar plot of Brady accuracy for different target categories"""
    sns.catplot(x="model", y="correct", hue="target_category", kind="bar",
                col="eval_type", ci=68, data=brady_results_summary_by_target_category)
    plt.show()

def get_target_category_word_frequency():
    """Get word frequency of Brady object categories from Multimodal-S training split."""
    with open('../data/train.json') as f:
        train = json.load(f)["data"]
     
    word_freq = {}
    target_categories = results_df['target_category'].unique().tolist()
    for utterance in train:
        for word in utterance["utterance"].split():
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
     
    # get word_freq for object categories and create a new dataframe
    word_freq_object_categories = {}
    for category in target_categories:
        word_freq_object_categories[category] = word_freq[category]
     
    word_freq_object_categories_df = {}
    word_freq_object_categories_df["target_category"] = list(word_freq_object_categories.keys())
    word_freq_object_categories_df["frequency"] = list(word_freq_object_categories.values())
    word_freq_object_categories_df = pd.DataFrame.from_dict(word_freq_object_categories_df, orient="columns")
    return word_freq_object_categories_df

def get_target_category_contextual_diversity():
    """Calculate contextual diversity for each target category"""
    with open('../data/vocab.json') as f:
        vocab = json.load(f)
     
    target_category_contextual_diversity = {}
    for target_category in target_categories:
        # get all utterances from train with target_category
        utterances = [utterance['utterance'] for utterance in train if target_category in utterance['utterance']]
        target_category_counts = np.zeros(len(vocab))
     
        # loop over utterances and get index of target_category in each utterance
        for utterance in utterances:
            words = utterance.split(' ')
            # get index of target_category in utterance, otherwise skip if not found
            try:
                target_category_index = words.index(target_category)
                # get adjacent indices to target_category_index
                context_utterance_idxs = [target_category_index - 1,
                    target_category_index + 1]
         
                # get vocab idxs of context words
                # return UNK if word not in vocab
                context_idxs = []
                for i, idx in enumerate(context_utterance_idxs):
                    if idx >= 0 and idx < len(words):
                        word = words[idx]
                        if word in vocab:
                            context_idxs.append(vocab[words[idx]])
                        else:
                            context_idxs.append(vocab['<unk>'])
                
                # increment target_category_counts by one for each context_idx
                for context_idx in context_idxs:
                    target_category_counts[context_idx] += 1
            except ValueError:
                continue
     
        # calculate entropy of target_category_counts
        target_category_entropy = entropy(target_category_counts)
        # get non-zero entries of target_category_counts
        target_category_nonzero_counts = target_category_counts[target_category_counts > 0]
        
        target_category_contextual_diversity[target_category] = target_category_entropy

    # convert target_category_contextual_diversity to dataframe
    target_category_contextual_diversity_df = {}
    target_category_contextual_diversity_df["target_category"] = list(target_category_contextual_diversity.keys())
    target_category_contextual_diversity_df["entropy"] = list(target_category_contextual_diversity.values())
    target_category_contextual_diversity_df = pd.DataFrame.from_dict(target_category_contextual_diversity_df, orient="columns")
    return target_category_contextual_diversity_df
    
def get_target_category_visual_diversity():
    # get visual diversity using CNN features
    # load pre-trained ResNet-50 model from pytorch and remove last layer to get features
    DATA_DIR = "/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal/train_5fps"
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(pretrained=True)
    layers = list(model.children())[:-1]
    model = torch.nn.Sequential(*layers).to(device)
    model.eval()
     
    target_category_visual_diversity = {}
    for target_category in target_categories:
        # get all utterances from train with target_category
        utterances = [utterance for utterance in train if target_category in utterance['utterance'].split(' ')]
         
        # get all frame_filenames from utterances and flatten
        frame_filenames = [utterance['frame_filenames'][0] for utterance in utterances]
         
        # get transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
         
        # read in images, convert to PIL and turn into a batch of tensors
        images = [Image.open(os.path.join(DATA_DIR, frame_filename)).convert('RGB') for frame_filename in frame_filenames]
        images = [transform(image) for image in images]
        images = torch.stack(images).to(device)
         
        # pass through model and get target_category_features
        with torch.no_grad():
            target_category_features = model(images)
            target_category_features = target_category_features.view(target_category_features.size(0), -1)
            print(target_category_features.size())
            target_category_features = target_category_features.cpu().numpy()
         
        # calculate standard deviation of target_category_features
        target_category_feature_std = np.mean(np.std(target_category_features, axis=0))
        target_category_visual_diversity[target_category] = target_category_feature_std
     
    # convert target_category_visual_diversity to dataframe
    target_category_visual_diversity_df = {}
    target_category_visual_diversity_df["target_category"] = list(target_category_visual_diversity.keys())
    target_category_visual_diversity_df["visual_diversity"] = list(target_category_visual_diversity.values())
    target_category_visual_diversity_df = pd.DataFrame.from_dict(target_category_visual_diversity_df, orient="columns")
    return target_category_visual_diversity_df

# # TODO: later combine all results
# # get summary results
# target_category_results = (results_original
#   >> group_by(_.target_category, _.eval_type, _.model)
#   >> summarize(correct_mean = _.correct.mean()))

# # join target category column
# target_category_results = pd.merge(target_category_results, word_freq_object_categories_df, on="target_category")

# # scatterplot in altair with frequency on the x-axis and correct_mean on the y-axis, facetted by model and eval_type
# base = alt.Chart().encode(
#     x=alt.X('frequency:Q', scale=alt.Scale(type='log')),
#     y=alt.Y('correct_mean:Q'),
#     color='model:N',
# )

# layers = alt.layer(base.mark_circle(), base.mark_text(
#     align='left', baseline='middle', dx=5).encode(
#     text='target_category'), base.transform_regression(
#         'frequency', 'correct_mean').mark_line(), data=target_category_results).facet(column='eval_type', row='model').interactive()
# save(layers, "../figures/eval_accuracy_word_freq_scatterplot.pdf")

# # calculate correlations
# target_category_results["log_frequency"] = np.log(target_category_results["frequency"])

# target_category_correlation = (target_category_results
#   >> group_by(_.eval_type, _.model)
#   >> summarize(cor = _.correct_mean.corr(_.log_frequency)))


# print(target_category_contextual_diversity)


# # join target category column to main results
# target_category_results = pd.merge(target_category_results, target_category_contextual_diversity_df, on="target_category")

# (target_category_results
#   >> summarize(cor = _.correct_mean.corr(_.entropy)))
# frequency_entropy_correlation


# target_category_results = pd.merge(target_category_results, target_category_visual_diversity_df, on="target_category")
    
# # get correlations
# (target_category_results
#   >> summarize(cor = _.correct_mean.corr(_.visual_diversity)))


if __name__ == "__main__":
    plot_saycam_summary_diff_init()

