
def test_step(model, data_loader, loss_fn, device):
    """
    Performs the testing step for the machine learning model

    Parameters
    ----------
    model: CNNModel instance
    data_loader: dataloader object, test data
    loss_fn: nn.CrossEntropyLoss instance, loss function
    device: str, device in which to train

    Returns
    -------
    test_loss: float, average testing loss for the dataloader at hand
    test_acc: float, average testing accuracy for the dataloader at hand
    """
    test_loss, test_acc = 0, 0
    model.eval()
    num_steps = 0

    with ((torch.inference_mode())):
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(target=y, preds=test_pred.argmax(dim=1)).item()
            num_steps += 1

        test_loss /= num_steps
        test_acc /= num_steps
        test_acc *= 100

    print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

    return test_loss, test_acc