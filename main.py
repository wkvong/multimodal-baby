import torch
import torchvision.models as models

# create mobilenetv2 architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier = torch.nn.Linear(in_features=1280, out_features=2765, bias=True)

# load in checkpoint
checkpoint = torch.load('models/TC-S.tar', map_location='cpu')

# rename checkpoint keys
prefix = 'module.'
n_clip = len(prefix)
renamed_checkpoint = {k[n_clip:]: v for k, v in checkpoint['model_state_dict'].items()}

# load state dict 
model.load_state_dict(renamed_checkpoint)

# remove classifier head
model = torch.nn.Sequential(*list(model.children())[:1])

x = torch.Tensor(1, 3, 224, 224)
print(x.size())

y = model(x)
print(y.size())
