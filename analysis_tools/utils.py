from pathlib import Path
import functools
import numpy as np
import torch


def frac_format(m, n, prec=2):
    return f'{m} / {n} = {m/n:.{prec}%}'


def identity(x):
    return x


def get_n_rows(n_items, n_cols):
    return (n_items - 1) // n_cols + 1


def get_np_attrs_from_values(values, attr):
    return np.array([getattr(value, attr) for value in values])


def get_tsne_points_from_vectors(
        vectors,
        n_components=2,
        random_state=0,
        perplexity=50,
        learning_rate='auto',
        n_iter=1000,
        metric='cosine',
        init='pca',
        **kwargs,
    ):
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        init=init,
        **kwargs,
    )
    points = tsne.fit_transform(vectors)
    print('T-SNE done.')
    return points

def get_eigen_points_from_vectors(vectors, centered=True, print_singular_values=False, **kwargs):
    from scipy.linalg import svd

    if centered:
        # subtract mean vector from vectors
        mean_vector = vectors.mean(0, keepdims=True)
        vectors = vectors - mean_vector

    U, s, Vh = svd(vectors, full_matrices=False, **kwargs)
    print('SVD done.')
    if print_singular_values:
        print('singular values:')
        print(s)
    return U

def get_pca_points_from_vectors(vectors, **kwargs):
    from sklearn.decomposition import PCA

    pca = PCA()

    reduced = pca.fit_transform(vectors)
    print('PCA done.')
    return reduced

def convert_attr_for_each(objs, get_attr='mean_vector', set_attr='tsne_point', converter=get_tsne_points_from_vectors, **kwargs):
    attrs = get_np_attrs_from_values(objs, get_attr)

    new_attrs = converter(attrs, **kwargs)

    for obj, new_attr in zip(objs, new_attrs):
        setattr(obj, set_attr, new_attr)

get_tsne_points = functools.partial(convert_attr_for_each, set_attr='tsne_point', converter=get_tsne_points_from_vectors)
get_eigen_points = functools.partial(convert_attr_for_each, set_attr='eigen_point', converter=get_eigen_points_from_vectors)
get_pca_points = functools.partial(convert_attr_for_each, set_attr='pca_point', converter=get_pca_points_from_vectors)


def torch_cache(cache_path):
    cache_path = Path(cache_path)

    def decorator(fn):
        def wrapper(*args, **kwargs):
            if cache_path.exists():
                # load from cache
                print(f'load from {cache_path}')
                data = torch.load(cache_path)

            else:
                data = fn(*args, **kwargs)
                # save to cache
                torch.save(data, cache_path)

            return data

        return wrapper

    return decorator


def get_model_device(model):
    return next(model.parameters()).device


default_value_formatter = lambda value: f'{value:5.3f}'
prob_formatter = lambda prob: f'{prob:6.1%}'


def print_top_values(values, idx2word, labels=None, top_k=5, steps=None,
                     value_formatter=default_value_formatter):
    """Print the top k words in values (optionally along with the labels)
    Inputs:
        values: a torch.Tensor of shape [n_steps, vocab_size] or [vocab_size]
        idx2word: mapping word index to word
        labels: a torch.Tensor of shape [n_steps] or []
        top_k: the number of top words to print
        steps: list of int, steps to print; None for all possible steps
        value_formatter: value_formatter(value) should get the formatted string
            of the value
    """

    # unsqueeze singleton inputs
    if values.dim() == 1:
        values = values.unsqueeze(0)
        if labels is not None:
            labels = labels.unsqueeze(0)

    # init default values
    if labels is None:
        labels = [None] * len(values)
    if steps is None:
        steps = list(range(len(labels)))

    top_values, top_indices = values.topk(top_k, -1)

    zipped = list(zip(values, labels, top_values, top_indices))
    for step in steps:
        value, label, top_value, top_index = zipped[step]
        formatter = lambda value, idx: f'{value_formatter(value)} {idx2word[idx]:8}'
        line = (formatter(value[label.item()].item(), label.item()) + ' | ' if label is not None else '') \
             + ' '.join(formatter(value.item(), index.item()) for value, index in zip(top_value, top_index))
        print(line)


def get_data_mean_img(data):
    from PIL import Image
    from torchvision import transforms
    from multimodal.multimodal_data_module import normalizer
    from multimodal.multimodal_saycam_data_module import EXTRACTED_FRAMES_DIRNAME
    import tqdm

    sum_example_img = 0

    to_tensor = transforms.ToTensor()

    for example in tqdm.tqdm(data):
        img_filenames = example["frame_filenames"]
        imgs = [
            to_tensor(Image.open(EXTRACTED_FRAMES_DIRNAME / img_filename).convert("RGB"))
            for img_filename in img_filenames]
        example_mean_img = sum(imgs) / len(imgs)

        sum_example_img += example_mean_img

    data_mean_img = sum_example_img / len(data)

    img = normalizer(data_mean_img)
    return img
