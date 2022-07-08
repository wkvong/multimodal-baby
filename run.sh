# full training
# python training/train.py --batch_size=32 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=True --pretrained_cnn --max_epochs=5 --multiple_frames --augment_frames

# fast dev run
python train.py --batch_size=16 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=False --pretrained_cnn --fast_dev_run --text_encoder="embedding" --shuffle_utterances

# python train.py --batch_size=8 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=False --pretrained_cnn --fast_dev_run --text_encoder="lstm"

