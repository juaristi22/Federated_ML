
def train_step(model, data_loader, loss_fn, optimizer, device):
    """
    Performs the training step for the machine learning model

    Parameters
    ----------
    model: CNNModel instance
    data_loader: dataloader object, train data
    loss_fn: nn.CrossEntropyLoss instance, loss function
    optimizer: torch,optim,SGD instance, learning optimizer
    device: str, device in which to train

    Returns
    -------
    train_loss: float, average training loss for the dataloader at hand
    train_acc: float, average training accuracy for the dataloader at hand
    """
    train_loss, train_acc = 0, 0
    model.train()
    num_steps = 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(target=y, preds=y_pred.argmax(dim=1)).item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        num_steps += 1

    train_loss /= num_steps
    train_acc /= num_steps
    train_acc *= 100

    return train_loss, train_acc