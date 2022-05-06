import torch
from torch import nn
import copy
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


from models.recurrent_autoencoder import RecurrentAutoencoder


class Autoencoder(object):
    def __init__(self, time_steps=100, embedding_dim=128, device=torch.device('cpu')):
        self.device = device
        self.time_steps = time_steps

        self.model = RecurrentAutoencoder(time_steps, 1, embedding_dim)
        self.model = self.model.to(self.device)

    def train(self, train_loader, val_loader, epochs=200, lr=0.0003, weight_decay=1e-4, save_path="../model", tensorboard_path="../runs", tensorboard_postfix="", log_steps=10):
        writer = SummaryWriter(os.path.join(tensorboard_path, tensorboard_postfix))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
        criterion = nn.L1Loss(reduction='sum').to(self.device)

        history = dict(train=[], val=[])

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 10000.0
        iter_count = 0

        for epoch in range(1, epochs + 1):
            # training
            self.model.train()
            train_losses = []
            for x_true, _ in train_loader:
                optimizer.zero_grad()

                x_true = x_true.to(self.device)
                x_pred = torch.transpose(self.model(x_true), 0, 1)

                # print(x_true.shape)
                # print(x_pred.shape)

                loss = criterion(x_pred, x_true)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                if(iter_count % log_steps == 0):
                    writer.add_scalar('loss', loss, global_step=iter_count)
                    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=iter_count)

                iter_count += 1

            # Warmup
            if epoch >= 10:
                scheduler.step()

            # validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for x_true, _ in val_loader:
                    x_true = x_true.to(self.device)
                    x_pred = torch.transpose(self.model(x_true), 0, 1)

                    loss = criterion(x_pred, x_true)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

            print(f'Epoch {epoch}: train loss {train_loss}, val loss {val_loss}')

        self.model.load_state_dict(best_model_wts)

        print("Training Complete")
        state = {
            'epoch': epochs,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        filename = f'checkpoint_{tensorboard_postfix}_{epochs}.pth.tar'
        torch.save(state, os.path.join(save_path, filename))
        print("Checkpoint Saved")

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def get_pred_and_loss(self, dataset):
        self.model.eval()
        preds = np.zeros((int(len(dataset)), self.time_steps), dtype='float32')
        losses = np.zeros((int(len(dataset)), self.time_steps), dtype='float32')
        # print(preds.shape)
        # print(losses.shape)

        for idx, (x_true, _) in enumerate(dataset):
            x_true = x_true.to(self.device)
            x_pred = torch.transpose(self.model(x_true), 0, 1)[0]
            loss = torch.abs(x_true-x_pred).detach().cpu().numpy()
            x_pred = x_pred.detach().cpu().numpy()
            
            preds[idx] = x_pred
            losses[idx] = loss

            # print(x_true)
            # print(x_pred)
            # print(loss)

        return preds, losses
