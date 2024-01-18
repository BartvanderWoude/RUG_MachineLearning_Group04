import datetime
import torch
from tqdm import tqdm

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for sample in tqdm(train_loader, position=0, leave=True):
            imgs = sample['image'].to(device)
            labels = sample['class'].to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)))