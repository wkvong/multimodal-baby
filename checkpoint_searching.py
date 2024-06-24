from pathlib import Path


checkpoint_dir = Path("/scratch/yq810/babyBerta/newBerta/saved_models")
for arch_dir in checkpoint_dir.rglob("lstm"):
    print(f'{arch_dir=}')
    for model_dir in arch_dir.iterdir():
        print(model_dir)
