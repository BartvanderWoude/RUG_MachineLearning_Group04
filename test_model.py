import torch
import torchvision
import os

from torch.utils.data import DataLoader

import code.dataset as dataset
import code.model as model
import code.test as test
import code.logger as logger

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    data = dataset.CBISDDSM(file="CBIS-DDSM/test.csv")

    data_loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=4)

    net = model.Net().to(device)
    net.load_state_dict(torch.load("output/models/model.pth"))
    loss_fn = torch.nn.CrossEntropyLoss()

    test.testing_loop(
        model = net,
        loss_fn = loss_fn,
        test_loader = data_loader,
        device=device
    )
    

if __name__ == '__main__':
    test_model()
