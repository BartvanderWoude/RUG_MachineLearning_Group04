import datetime
import torch
from tqdm import tqdm
import os

def testing_loop(model, loss_fn, test_loader, device):
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in tqdm(test_loader, position=0, leave=True):
            imgs = sample['image'].to(device)
            labels = sample['class'].to(device) 

            outputs = model(imgs)
            loss = loss_fn(outputs.view(-1,2), labels)

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            correct = correct + torch.eq(torch.argmax(outputs, dim=1), labels).sum().item()
            total = total + labels.shape[0]
            
            loss_val += loss.item()
    
    print('{} Test loss {}, Accuracy {}'.format(
        datetime.datetime.now(), loss_val / len(test_loader), correct / total))
    
    if not os.path.exists("output/logs"):
        os.makedirs("output/logs")

    with open("output/logs/test_loss.csv", "w") as f:
        f.write("loss\n")
        f.write(str(loss_val / len(test_loader)) + "\n")