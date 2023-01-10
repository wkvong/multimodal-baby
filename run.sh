# full training
# python training/train.py --batch_size=32 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=True --pretrained_cnn --max_epochs=5 --multiple_frames --augment_frames

# fast dev run
# standard model
# python train.py --batch_size=16 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=False --pretrained_cnn --multiple_frames --augment_frames --fast_dev_run --text_encoder="embedding" 

# allow fine-tuning
# python train.py --batch_size=16 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=False --pretrained_cnn --finetune_cnn --multiple_frames --augment_frames --fast_dev_run --text_encoder="embedding" 

# random init
python train.py --batch_size=8 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=False --multiple_frames --fast_dev_run --text_encoder="embedding" 
