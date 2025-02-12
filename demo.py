import torch
from multimodal.multimodal_lit import MultiModalLitModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cvcl, preprocess = MultiModalLitModel.load_model(model_name="cvcl")
cvcl = cvcl.to(device)
cvcl.eval()

# create random image to encode
images = torch.rand(4, 3, 224, 224).to(device)
image_features = cvcl.encode_image(images)

# create texts to encode
texts = ["ball", "puzzle", "car"]
texts, texts_len = cvcl.tokenize(texts)
texts, texts_len = texts.to(device), texts_len.to(device)
texts_features = cvcl.encode_text(texts, texts_len)

# get logits from a batch of images and texts
logits_per_image, logits_per_text = cvcl(images, texts, texts_len)

print("Logits per image shape:", logits_per_image.shape)
print("Logits per text shape:", logits_per_text.shape)
