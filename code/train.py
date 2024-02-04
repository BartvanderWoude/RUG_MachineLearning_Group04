import datetime
import torch
from tqdm import tqdm

def training_loop(n_epochs, optimizer, model, loss_fn, log, train_loader, validation_loader, device):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for sample in tqdm(train_loader, position=0, leave=True):
            imgs = sample['image'].to(device)
            labels = sample['class'].to(device)
            
            outputs = model(imgs)
            loss = loss_fn(outputs.view(-1) , labels.to(torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
        
        log.log_training_loss(epoch, loss_train / len(train_loader))
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)))
        
        loss_val = 0.0
        correct = 0
        for sample in tqdm(validation_loader, position=0, leave=True):
            imgs = sample['image'].to(device)
            labels = sample['class'].to(device)            

            outputs = model(imgs)
            correct = correct + torch.eq(torch.argmax(outputs, dim=1), labels).sum().item()
            loss = loss_fn(outputs.view(-1), labels.to(torch.float32))
            
            loss_val += loss.item()
        
        log.log_validation_loss(epoch, loss_val / len(validation_loader))
        print('{} Epoch {}, Validation loss {}, Accuracy {}'.format(
            datetime.datetime.now(), epoch, loss_val / len(validation_loader), correct))