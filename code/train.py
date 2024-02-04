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

def training_loop(n_epochs, optimizer, model, loss_fn, log, train_loader, validation_loader, device):
    ## Training loop
    previous_val_loss = float('inf')
    earlystop = 0
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        loss_train = 0.0
        for sample in tqdm(train_loader, position=0, leave=True):
            imgs = sample['image'].to(device)
            labels = sample['class'].to(device)
            
            outputs = model(imgs)
            loss = loss_fn(outputs.view(-1,2) , labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
        
        log.log_training_loss(epoch, loss_train / len(train_loader))
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)))
        
        # Validation
        model.eval()
        loss_val = 0.0
        correct = 0
        total = 0

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        with torch.no_grad():
            for sample in tqdm(validation_loader, position=0, leave=True):
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
        
        log.log_validation_loss(epoch, loss_val / len(validation_loader), correct / total, f1score(tp, tn, fp, fn), fnr(tp, tn, fp, fn))
        print('{} Epoch {}, Validation loss {}, Accuracy {}, F1-score {}, FNR {}'.format(
            datetime.datetime.now(), epoch, loss_val / len(validation_loader), correct / total, f1score(tp, tn, fp, fn), fnr(tp, tn, fp, fn)))
        
        # Early stopping
        if (loss_val / len(validation_loader)) < previous_val_loss:
            previous_val_loss = loss_val / len(validation_loader)
            if not os.path.exists("output/models"):
                os.makedirs("output/models")
            torch.save(model.state_dict(), "output/models/model-%s.pth" % (str(epoch)))
            earlystop = 0
        else:
            print("Early stopping")
            earlystop = earlystop + 1
            if earlystop == 5:
                break
    return