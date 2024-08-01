import torch
from torch import nn

class CNNModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 14 * 14, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.classifier(x)
        return x

class NewModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 8 * 8, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


class Client(FM.NewModel):
    def __init__(self, input_shape, hidden_units, output_shape,
                 epochs=None, data=None, learning_rate=0.000001, device=device):
        super().__init__(input_shape, hidden_units, output_shape)
        self.lr = learning_rate
        self.optimizer = None
        self.epochs = epochs
        self.data = data
        self.parent = None
        self.name = None

    def train_step(self, data_loader, loss_fn, optimizer, device=device):
        for epoch in range(self.epochs):
            train_loss, train_acc = 0, 0

            self.train()
            num_steps = 0

            for batch, (X, y) in enumerate(data_loader):
                X, y = X.to(device), y.to(device)
                y_pred = self.forward(X)
                loss = loss_fn(y_pred, y)
                train_loss += loss.item()
                train_acc += FM.accuracy_fn(target=y, preds=y_pred.argmax(dim=1)).item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                num_steps += 1

            train_loss /= num_steps
            train_acc /= num_steps
            train_acc *= 100

        # print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

        return train_loss, train_acc

    def test_step(self, data_loader, loss_fn, device=device):
        test_loss, test_acc = 0, 0
        self.eval()
        num_steps = 0

        with ((torch.inference_mode())):
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                test_pred = self.forward(X)
                loss = loss_fn(test_pred, y)
                test_loss += loss.item()
                test_acc += FM.accuracy_fn(target=y, preds=test_pred.argmax(dim=1)).item()
                num_steps += 1

            test_loss /= num_steps
            test_acc /= num_steps
            test_acc *= 100

        print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

        return test_loss, test_acc
    def named(self, n):
        self.name = (f"{self.__class__.__name__}_{n}")
    def __str__(self):
        return self.name

class Aggregator(FM.NewModel):
    def __init__(self, input_shape, hidden_units, output_shape, device=device):
        super().__init__(input_shape, hidden_units, output_shape)
        self.parent = None
        self.children_nodes = []
        self.name = None

    def add_child(self, child):
        self.children.append(child)
    def test_step(self, data_loader, loss_fn, device=device):
        test_loss, test_acc = 0, 0
        self.eval()
        num_steps = 0

        with ((torch.inference_mode())):
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                test_pred = self.forward(X)
                loss = loss_fn(test_pred, y)
                test_loss += loss.item()
                test_acc += FM.accuracy_fn(target=y, preds=test_pred.argmax(dim=1)).item()
                num_steps += 1

            test_loss /= num_steps
            test_acc /= num_steps
            test_acc *= 100

        print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")

        return test_loss, test_acc

    def average(self, clients_params):
        with torch.no_grad():
            averaged_params = clients_params.popitem()[1]
            for parameter in averaged_params:
                for model_state in clients_params:
                    parameter_value = clients_params[model_state][parameter]
                    parameter_value += averaged_params[parameter]
                    parameter_value = (1 / 2) * torch.clone(parameter_value)
                    averaged_params[parameter] = parameter_value
        return averaged_params

    def named(self, n):
        self.name = (f"{self.__class__.__name__}_{n}")

    def __str__(self):
        return self.name