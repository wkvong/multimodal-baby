from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule
from multimodal.coco_captions_data_module import COCOCaptionsDataModule
from train import _setup_parser



DataModuleClasses = {
    "saycam": MultiModalSAYCamDataModule,
    "coco": COCOCaptionsDataModule,
}


def build_data(args=None, dataset=None, deterministic=True, return_args=False):
    """Build data and vocab.
    Input:
        args: args from _setup_parser() in train.py
        dataset: can be a MultiModalDataModule class, a string, or None;
            if None, use args.dataset
        deterministic: make the train dataloader deterministic
        return_args: whether to return args
    Returns:
        data: instance of MultiModalDataModule
        args: if return_args
    """

    # parse empty args
    parser = _setup_parser()
    args_ = parser.parse_args("")
    # set args
    if args is not None:
        for key, value in args.items():
            setattr(args_, key, value)
    args = args_
    # make the train dataloader deterministic
    if deterministic:
        args.augment_frames = False
        args.multiple_frames = False
        args.multiple_captions = False
        args.eval_include_sos_eos = True

    # build data module
    if dataset is None:
        dataset = getattr(args, "dataset", "saycam")
    if isinstance(dataset, str):
        dataset = DataModuleClasses[dataset]
    data = dataset(args)
    data.prepare_data()
    data.setup()

    if return_args:
        return data, args
    else:
        return data
