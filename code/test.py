import datetime
import torch
from tqdm import tqdm
import os

def f1score(tp, tn, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def fnr(tp, tn, fp, fn):
    return fn / (tp + fn)

def testing_loop(model, loss_fn, test_loader, device):
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for sample in tqdm(test_loader, position=0, leave=True):
            imgs = sample['image'].to(device)
            labels = sample['class'].to(device) 

            outputs = model(imgs)
            loss = loss_fn(outputs.view(-1,2), labels)

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            correct = correct + torch.eq(torch.argmax(outputs, dim=1), labels).sum().item()
            total = total + labels.shape[0]

            if torch.argmax(outputs, dim=1) == 1:
                if labels.item() == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if labels.item() == 0:
                    tn = tn + 1
                else:
                    fn = fn + 1
            
            loss_val += loss.item()
    
    print('{} Test loss {}, Accuracy {}, F1-score {}, FNR {}'.format(
        datetime.datetime.now(), loss_val / len(test_loader), correct / total, f1score(tp, tn, fp, fn), fnr(tp, tn, fp, fn)))
    
    if not os.path.exists("output/logs"):
        os.makedirs("output/logs")

    with open("output/logs/test_loss.csv", "w") as f:
        f.write("loss,accuracy,f1-score,fnr\n")
        f.write(str(loss_val / len(test_loader)) + "," 
                + str(correct/total) + "," 
                + str(f1score(tp,tn,fp,fn)) + "," 
                + str(fnr(tp,tn,fp,fn)) + "\n")