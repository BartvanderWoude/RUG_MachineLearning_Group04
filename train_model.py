import torch
import torchvision
import os

from torch.utils.data import DataLoader

import code.dataset as dataset
import code.model as model
import code.train as train
import code.logger as logger

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    data = dataset.CBISDDSM(file="CBIS-DDSM/train-augmented.csv")
    valdata = dataset.CBISDDSM(file="CBIS-DDSM/val.csv")
    # train_size = int(0.8 * len(data))
    # validation_size = len(data) - train_size
    # train_dataset, validation_dataset = torch.utils.data.random_split(data, [train_size, validation_size])

    data_loader_train = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)
    data_loader_validation = DataLoader(valdata, batch_size=1, shuffle=True, num_workers=4)

    net = model.Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    log = logger.Logger()

    train.training_loop(
        n_epochs = 50,
        optimizer = optimizer,
        model = net,
        loss_fn = loss_fn,
        log = log,
        train_loader = data_loader_train,
        validation_loader = data_loader_validation,
        device=device
    )

if __name__ == '__main__':
    train_model()
