import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_dwsnets(model, loader, device=torch.device("cuda")):
    '''
    Evaluate function for DWSNets model

    Args:
        model (nn.Module): The dwsnets model to evaluate.
        loader (DataLoader): The dataloader for the dataset.
        device (str): The device to run the evaluation on.
    
    Returns:
        dict: A dictionary containing the average loss(avg_loss), average accuracy(avg_acc), predicted labels(predicted) and ground truth labels(gt).
    '''
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    for batch in loader:
        batch = batch.to(device)
        inputs = (batch.weights, batch.biases)
        out = model(inputs)
        loss += F.cross_entropy(out, batch.label, reduction="sum")
        total += len(batch.label)
        pred = out.argmax(1)
        label = batch.label.argmax(1)
        correct += pred.eq(label).sum()
        predicted.extend(pred.cpu().numpy().tolist())
        gt.extend(batch.label.cpu().numpy().tolist())

    model.train()
    avg_loss = loss / total
    avg_acc = correct / total

    return dict(avg_loss=avg_loss, avg_acc=avg_acc, predicted=predicted, gt=gt)

def get_accuracy_st(model, dataloader, emb=None):
    '''
        Get the accuracy of a Set Transformer model on the dataloader.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_correct = 0
    model.eval()
    for X, y in dataloader:
        X = X.unsqueeze(2)
        if(emb is not None):
            emb_batch = emb.repeat(X.shape[0], 1, 1)
            X = torch.cat([emb_batch, X], dim=2)
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X.float())
        # Accuracy
        y = torch.argmax(y, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        correct = (y_pred == y).sum()
        total_correct += correct

    accuracy_trained = total_correct / len(dataloader.dataset) * 100
    return accuracy_trained.item()

def get_accuracy_mlp(model, dataloader, device=None):
    if(device is None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_correct = 0
    model.eval()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X.float())
        # Accuracy
        y = torch.argmax(y, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        correct = (y_pred == y).sum()
        total_correct += correct

    accuracy_trained = total_correct / len(dataloader.dataset) * 100
    return accuracy_trained.item()
