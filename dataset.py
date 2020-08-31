import os
import shutil
import glob
from tqdm import tqdm
import numpy as np

def generate_eval_frames():
    # set random seed
    np.random.seed(0)

    # directories
    original_eval_dir = '/Users/waikeenvong/code/multimodal-baby/data/evaluation/original/'
    new_val_dir = '/Users/waikeenvong/code/multimodal-baby/data/evaluation/val/'
    new_test_dir = '/Users/waikeenvong/code/multimodal-baby/data/evaluation/test/'
    
    eval_categories = os.listdir(original_eval_dir)
    for eval_category in eval_categories:
        eval_category_dir = os.path.join(original_eval_dir, eval_category)
        eval_category_frames = sorted(os.listdir(eval_category_dir))

        # get indices to split original eval dataset into val and test
        split_idxs = np.arange(len(eval_category_frames))
        np.random.shuffle(split_idxs)
        val_idxs = split_idxs[:int(len(eval_category_frames) * 0.2)]
        test_idxs = split_idxs[int(len(eval_category_frames) * 0.2):]

        # check dataset has been split correctly
        assert len(val_idxs) + len(test_idxs) == len(split_idxs)

        # copy over validation frames to new directory
        print(f'copying {eval_category} frames for validation set')
        for val_idx in tqdm(val_idxs):
            # get path to original frame
            original_filename = os.path.join(original_eval_dir, eval_category, eval_category_frames[val_idx])

            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(new_val_dir, eval_category)):
                os.makedirs(os.path.join(new_val_dir, eval_category))

            # copy frame
            shutil.copyfile(original_filename, os.path.join(new_val_dir, eval_category, eval_category_frames[val_idx]))

        # copy over test frames to new directory
        print(f'copying {eval_category} frames for test set')
        for test_idx in tqdm(test_idxs):
            # get path to original frame
            original_filename = os.path.join(original_eval_dir, eval_category, eval_category_frames[test_idx])

            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(new_test_dir, eval_category)):
                os.makedirs(os.path.join(new_test_dir, eval_category))

            # copy frame
            shutil.copyfile(original_filename, os.path.join(new_test_dir, eval_category, eval_category_frames[test_idx]))

            
if __name__ == "__main__":
    generate_eval_frames()
