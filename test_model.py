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

    # Note: for tp, tn, fp, fn calculation, batch_size must be 1
    data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=4)

    net = model.Net().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    files = os.listdir("output/models/")
    files = [x.split(".")[0] for x in files]
    files = [x.split("-")[1] for x in files if len(x.split("-")) > 1]
    highest = 0
    for x in range(50):
        if str(x) in files:
            highest = x
    print("output/models/model-" + str(highest) + ".pth")
    net.load_state_dict(torch.load("output/models/model-" + str(highest) + ".pth"))

    test.testing_loop(
        model = net,
        loss_fn = loss_fn,
        test_loader = data_loader,
        device=device
    )
    

if __name__ == '__main__':
    test_model()
