import torch

def score(model, dataloader, device):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            y_pred.extend(torch.argmax(out, dim=-1).tolist())
            y_true.extend(y.tolist())
    return y_true, y_pred