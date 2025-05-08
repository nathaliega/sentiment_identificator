import torch

def train(dataloader, model, optimizer, loss_fn):
    """Trains the model for one epoch using the provided data loader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of (text, label, length).
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        loss_fn (Callable): Loss function to compute the training loss.

    Returns:
        Tuple[float, float]: A tuple containing:
            - average accuracy over the dataset (float)
            - average loss over the dataset (float)
    """
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), \
    total_loss/len(dataloader.dataset)


def evaluate(dataloader, model, loss_fn):
    """Evaluates the model using the provided data loader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of (text, label, length).
        model (torch.nn.Module): The model to be evaluated.
        loss_fn (Callable): Loss function to compute the evaluation loss.

    Returns:
        Tuple[float, float]: A tuple containing:
            - average accuracy over the dataset (float)
            - average loss over the dataset (float)
    """
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)