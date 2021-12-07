import torch.nn.functional as F

def get_entropy(logits, dim=-1):
    log_p = F.log_softmax(logits, dim=dim)
    return (F.softmax(log_p, dim=dim) * -log_p).sum(dim=dim) # E[- log p] = sum - p log p
