import cProfile
import gc

import pytorch_lightning as pl
import torch
import tqdm
import os
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models


def collate_fn(batch):
    batch_size = sum([1 for b in batch if b is not None])
    max_options = max([len(b[2]) for b in batch if b is not None])
    anchors = torch.empty((batch_size,20,90,8,8))
    positives = torch.empty((batch_size,12,8,8))
    negative = torch.zeros((batch_size,12,8,8))
    encodings = torch.empty((batch_size,50,8,8))
    i = 0
    for batch_index in range(len(batch)):
        if batch[batch_index] is None or batch[batch_index][3] == 0:
            raise Exception('Batch is None')
        padded_anchor = torch.zeros((20,90,8,8))
        len_unpadded_anchor = batch[batch_index][0].size(0)
        padded_anchor[:len_unpadded_anchor,:,:,:] = batch[batch_index][0]
        player_encoding = batch[batch_index][4]

        anchors[i,:,:,:,:] = padded_anchor
        positives[i,:,:,:] = batch[batch_index][1]
        negative[i,:,:,:] = batch[batch_index][2]
        encodings[i,:,:,:] = player_encoding
        i += 1
    return anchors,positives,negative,encodings

def validation_loop(network,val_loader,epoch):
    with torch.no_grad():
        network.eval()
        bar = tqdm.tqdm(enumerate(val_loader),\
                total = len(val_loader), mininterval = 1, desc = 'Validation')
        for i,batch in bar:
            if epoch == 0 and i >= len(val_loader)//10:
                break
            loss = network.validation_step((batch),i)
            loss = loss.item()
            network.eval_losses.append(loss)
            if i % 10 == 0:
                network.writer.add_scalar('eval_loss',loss,epoch*len(val_loader)+i)
                bar.set_postfix({'Loss': loss})
        return network.validation_epoch_end(epoch)

    
def training_loop(network):
    #hyperparameters
    patience = 3
    persistent_workers = False
    pin_memory = True
    shuffle_dataset = False


    network = network.to(network.device)
    val_data = models.Siamese_RBC_dataset(val_path,shuffle_on_init= shuffle_dataset,num_choices = None,max_samples = val_max_samples)
    val_loader = DataLoader(val_data,batch_size,shuffle = False, persistent_workers=persistent_workers, pin_memory = pin_memory,collate_fn= collate_fn, num_workers=num_workers)
    train_data = models.Siamese_RBC_dataset(train_path, shuffle_on_init=shuffle_dataset,num_choices = None,max_samples = max_samples)
    train_loader = DataLoader(train_data,batch_size,shuffle = True, persistent_workers=persistent_workers, pin_memory = pin_memory, collate_fn= collate_fn, num_workers=num_workers)

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_losses.append(validation_loop(network,val_loader,0))
    epoch = 0
    no_improvement = 0
    while True:
        network.train()

        bar = tqdm.tqdm(enumerate(train_loader),\
                total = len(train_loader), mininterval = 1, desc = 'Training')
        for i,batch in bar:
            loss = network.training_step(batch,i)
            loss = loss.item()
            network.train_losses.append(loss)
            if i % 10 == 0:
                network.writer.add_scalar('train_loss',loss,epoch*len(train_loader)+i)
                bar.set_postfix({'Loss': loss})
            if i % 1_000 == 0:
                torch.save(network.state_dict(),'Prediction_RBC_WIP.pt')


        new_loss = network.training_epoch_end(epoch)
        epoch_train_losses.append(new_loss)
        print(f'Training loss for epoch {epoch}: {new_loss}')

        
        new_loss = validation_loop(network,val_loader, epoch +1)
        #Check for early stopping
        if len(epoch_val_losses) != 0 and new_loss >= min(epoch_val_losses):
            print('No improvement for this epoch')
            no_improvement += 1
            if no_improvement == patience:
                print('Early stopping')
                break
        else:
            torch.save(network.state_dict(),'Prediction_RBC.pt')
        epoch_val_losses.append(new_loss)
        print(f'Validation loss for epoch {epoch}: {new_loss}')

        if no_improvement == patience:
            break
        epoch += 1
        gc.collect()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    batch_size = 1024
    num_workers = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    #create network
    network = models.Probability_Network(create_writer = False, device = device)
    network.optimizer = torch.optim.AdamW(network.parameters(), lr = 3e-4)

    #create data
    max_samples = None
    val_max_samples = int(max_samples*0.1) if max_samples is not None else None
    train_path = '../RBC/data/siamese_playerlabel/train/'
    val_path = '../RBC/data/siamese_playerlabel/val/'

    cProfile.run('training_loop(network)', sort = 'tottime')
    training_loop(network)
