import torch
import torchvision

from torch.utils.data import DataLoader

import code.dataset as dataset
import code.model as model
import code.train as train
import code.logger as logger

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    data = dataset.CBISDDSM(file="CBIS-DDSM/train-augmented.csv")
    train_size = int(0.8 * len(data))
    validation_size = len(data) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(data, [train_size, validation_size])

    data_loader_train = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    data_loader_validation = DataLoader(validation_dataset, batch_size=4, shuffle=True, num_workers=4)

    net = model.Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-5)
    loss_fn = torch.nn.BCELoss()
    log = logger.Logger()

    train.training_loop(
        n_epochs = 20,
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
