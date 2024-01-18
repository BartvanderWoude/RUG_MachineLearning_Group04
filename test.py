import torch
import torchvision

from torch.utils.data import DataLoader

import code.dataset as dataset
import code.model as model
import code.train as train

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    data = dataset.CBISDDSM()
    data_loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=4)

    net = model.Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
    loss_fn = torch.nn.CrossEntropyLoss()

    train.training_loop(
        n_epochs = 100,
        optimizer = optimizer,
        model = net,
        loss_fn = loss_fn,
        train_loader = data_loader,
        device=device
    )

if __name__ == '__main__':
    test()
