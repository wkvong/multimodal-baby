from multimodal.attention_maps import Hook, gradCAM_with_act_and_grad, n_inv, plot_image
from .utils import get_model_device


def torch_to_numpy_image(img):
    return img.permute(1, 2, 0).cpu().numpy()


def gradCAM_for_captioning_lm(model, x, y, y_len, steps=None):
    if steps is None:
        steps = list(range(y_len.item()))

    device = get_model_device(model)

    model.language_model.text_encoder.lstm.train()

    layer = model.vision_encoder.model.layer4

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        loss, outputs, logits, attns, labels = model.calculate_ce_loss(
            y.unsqueeze(0).to(device), y_len.unsqueeze(0).to(device), x=x.unsqueeze(0).to(device),
            tokenwise=True)

        gradcams = []
        for step in steps:
            if step == 0:
                gradcam = None
            else:
                hook.data.grad = None
                loss[0, step - 1].backward(retain_graph=True)
                gradcam = gradCAM_with_act_and_grad(hook.activation, -hook.gradient)
                gradcam = gradcam.squeeze().detach().cpu().numpy()
            gradcams.append(gradcam)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    model.language_model.text_encoder.lstm.eval()

    return gradcams


def attention_for_attention_lm(model, x, y, y_len, steps=None):
    if steps == list(range(y_len.item())):
        steps = None

    device = get_model_device(model)

    loss, outputs, logits, attns, labels = model.calculate_ce_loss(
        y.unsqueeze(0).to(device), y_len.unsqueeze(0).to(device), x=x.unsqueeze(0).to(device))

    ret = attns[0].detach().cpu().numpy()
    return ret if steps is None else ret[steps]
