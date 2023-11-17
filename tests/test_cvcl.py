import glob
import torch

from multimodal.multimodal_lit import MultiModalLitModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cvcl, preprocess = MultiModalLitModel.load_model(model_name="cvcl")
cvcl = cvcl.to(device)
cvcl.eval()
print("CVCL model loaded!")

# create random image tensor
images = torch.rand(4, 3, 224, 224).to(device)
image_features = cvcl.encode_image(images)
print("image features:", image_features.size())

# create random text tensor
# texts = ["ball", "puzzle", "car"]
texts = ["ball"]
texts, texts_len = cvcl.tokenize(texts)
texts, texts_len = texts.to(device), texts_len.to(device)
texts_features = cvcl.encode_text(texts, texts_len)
print("text features:", texts_features.size())

# test model
logits_per_image, logits_per_text = cvcl(images, texts, texts_len)
print("logits per image:", logits_per_image.size())
print("logits per text:", logits_per_text.size())
