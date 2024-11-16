import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data.dataloader
import torch.utils.data.dataset

class KryptonitePipeline():
    def __init__(self, model, n, batch_size, optimizer, loss_fn, epochs, device):
        self.n = n
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.n_epochs = epochs
        self.device = device

    def load_data(self):
        x_file_path = 'Datasets/kryptonite-' + str(self.n) + '-X.npy'
        y_file_path = 'Datasets/kryptonite-' + str(self.n) + '-y.npy'

        x_raw = torch.tensor(np.load(x_file_path), dtype=torch.float32)
        y_raw = torch.tensor(np.load(y_file_path), dtype=torch.float32)
        row_count = x_raw.shape[0]
        print(f"Row count: {row_count}")

        X_train, X_val = torch.tensor_split(
            x_raw,
            [round(row_count * 0.8)],
            dim=0
        )

        y_train, y_val = torch.tensor_split(
            y_raw,
            [round(row_count * 0.8)],
            dim=0
        )

        train_dataset = torch.utils.data.TensorDataset(
            X_train.to(self.device),
            y_train.to(self.device)
        )
        val_dataset = torch.utils.data.TensorDataset(
            X_val.to(self.device),
            y_val.to(self.device)
        )

        print(len(X_val))

        self.loaders = {
            'train': torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            ),
            'validation': torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )
        }

    def train_model(self):
        for epoch in range(self.n_epochs):
            loss_arr = []
            for data in self.loaders['train']:
                x_vals, labels = data
                self.optimizer.zero_grad()
                label_pred = self.model(x_vals)
                labels = torch.reshape(labels, (self.batch_size, 1))
                loss = self.loss_fn(label_pred, labels)
                loss.backward()
                self.optimizer.step()
                loss_arr.append(loss.item())
            
            print(f"Finished epoch {epoch + 1}, epoch loss {np.average(loss_arr)}")

    def evaluate_model(self, val: bool):
        accuracy = np.array([])
        with torch.no_grad():
            if val:
                loader_to_use = 'validation'
            else:
                loader_to_use = 'train'
            for data in self.loaders[loader_to_use]:
                x, labels = data
                output = self.model(x)
                output = output.cpu()
                labels = labels.cpu()
                accuracy = np.concatenate((accuracy, torch.eq(torch.flatten(output.round()), labels).numpy()))

        accuracy = accuracy.mean()

        if val:
            print(f"Validation accuracy is: {accuracy}")
        else:
            print(f"Train accuracy is: {accuracy}")

        return accuracy

class KryptoniteModel(nn.Module):
    def __init__(self, structure):
        super().__init__()
        self.layer1 = nn.Linear(structure[0], structure[1])
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(structure[1], structure[2])
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(structure[2], structure[3])
        self.act3 = nn.ReLU()
        self.output = nn.Linear(structure[3], structure[4])
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.act_output(self.output(x))
        
        return x


if __name__ == "__main__":
    accuracy_thresholds = {9: 94.8, 12: 92.3, 15: 89.8, 18: 87.3, 24: 79.8, 30: 74.8, 45: 69.8}
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    n = 30
    structure = [n, 42, 18, 6, 1]
    batch_size = 100
    lr = 0.07
    epochs = 10000
    loss_function = torch.nn.BCELoss()
    model = KryptoniteModel(structure)
    model.to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    pipeline = KryptonitePipeline(
        model=model,
        n=n,
        loss_fn=loss_function,
        batch_size=batch_size,
        optimizer=optimizer,
        epochs=epochs,
        device=device
    )

    pipeline.load_data()
    pipeline.train_model()
    train_acc = pipeline.evaluate_model(False)
    val_acc = pipeline.evaluate_model(True)

    threshold = accuracy_thresholds[n]

    if train_acc and val_acc > threshold:
        print(f"Accuracy is above set threshold, saving weights: ")
        model_dict_file_name = f"model_dict_{n}_{'-'.join(map(str, structure))}_{batch_size}_{lr}_{epochs}.pth"
        model_full_file_name = f"model_full_{n}_{'-'.join(map(str, structure))}_{batch_size}_{lr}_{epochs}.pth"