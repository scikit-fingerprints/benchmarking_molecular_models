import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Classifier, Xtransformer_Encoder
from losses import ContrastiveLoss, Optimizer, Scheduler, RMSELoss
from sklearn.metrics import roc_auc_score, mean_squared_error
from dataloader import GenerateDataset, ScaffoldSplitter
from pickle import dump


class Trainer(nn.Module):
    def __init__(self, args, data):
        super(Trainer, self).__init__()
        self.args = args
        self.data = data

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load dataset
        if args.task == 'pretraining':
            path = 'data/embedding/pubchem_part.pth'
        elif args.task == 'downstream' or 'inference':
            path = f'data/prediction/{data}.pth'

        smiles, second = torch.load(path)

        if data == 'hiv' or data == 'bace' or data == 'bbbp':   # scaffold split
            train_idx, valid_idx, test_idx = ScaffoldSplitter(data=data, seed=args.seed).scaffold_split()
            train_dataset = GenerateDataset(args, smiles, second, phase='train', train_idx=train_idx,
                                            valid_idx=valid_idx, test_idx=test_idx)
            valid_dataset = GenerateDataset(args, smiles, second, phase='valid', train_idx=train_idx,
                                            valid_idx=valid_idx, test_idx=test_idx)
            test_dataset = GenerateDataset(args, smiles, second, phase='test', train_idx=train_idx,
                                           valid_idx=valid_idx, test_idx=test_idx)

            self.test_iter = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
            print(f'Test dataset size: {len(test_dataset)}')
        else:
            if args.task == 'downstream':
                test_dataset = GenerateDataset(args, smiles, second, phase='test')
                self.test_iter = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
                print(f'Test dataset size: {len(test_dataset)}')

            train_dataset = GenerateDataset(args, smiles, second, phase='train')
            valid_dataset = GenerateDataset(args, smiles, second, phase='valid')

        self.train_iter = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.valid_iter = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f'Train dataset size: {len(train_dataset)}')
        print(f'Valid dataset size: {len(valid_dataset)}')

        # define model
        self.model = Xtransformer_Encoder(args).to(self.device)

        # loss
        if args.criterion == 'contrastive':
            self.criterion = ContrastiveLoss(temperature=args.temperature)
        elif args.criterion == 'rmse':
            self.criterion = RMSELoss()
        elif args.criterion == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()

        # optimizer and scheduler
        self.optimizer = Optimizer(args, self.model)
        self.scheduler = Scheduler(args, self.optimizer)

    def train(self):
        start = time.time()

        # early stopping
        best_epoch = 0
        best_loss = np.inf
        early_stopping = 0

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_one_epoch()

            self.scheduler.step()

            val_loss = self.validate()

            print(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            # save the model for every epoch
            os.makedirs('models/pretrained', exist_ok=True)
            save_path = f'models/pretrained/pre_model_epoch_{epoch}.pth'
            torch.save(self.model.state_dict(), save_path)

            if val_loss < best_loss:
                early_stopping = 0
                best_epoch = epoch
                best_loss = val_loss
                torch.save(self.model.state_dict(), 'models/pretrained/pretrained_best_model.pth')
                print('Saved the best model')

            else:
                early_stopping += 1

            if early_stopping == self.args.patience:
                break

        end = time.time()
        print(f'Best Epoch: {best_epoch} | Best Val Loss: {best_loss:.4f}')
        print(f'Total Training time:{(end - start) / 60:.3f} minutes')

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0

        for smiles, second in tqdm(self.train_iter):
            smiles, second = smiles.to(self.device), second.to(self.device)

            self.optimizer.zero_grad()

            smiles_output = self.model(smiles)
            second_output = self.model(second)

            loss = self.criterion(smiles_output, second_output)
            loss.requires_grad_(True)
            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(self.train_iter)

        return train_loss

    def validate(self):
        self.model.eval()

        with torch.no_grad():
            val_loss = 0

            for smiles, second in tqdm(self.valid_iter):
                smiles, second = smiles.to(self.device), second.to(self.device)

                smiles_output = self.model(smiles)
                second_output = self.model(second)

                loss = self.criterion(smiles_output, second_output)
                val_loss += loss.item()

            val_loss /= len(self.valid_iter)

        val_loss /= len(self.valid_iter)

        return val_loss

    def pre_train(self):
        self.model.load_state_dict(torch.load('models/pretrained/pretrained_best_model.pth'))
        self.model = Classifier(self.args, encoder=self.model, num_classes=self.args.num_classes).to(self.device)

        start = time.time()

        # early stopping
        best_loss = np.inf
        early_stopping = 0

        # save path
        os.makedirs('models/downstream', exist_ok=True)

        for epoch in range(self.args.epochs):
            train_loss = self.pre_train_one_epoch()
            val_loss, metric = self.pre_validate()      # return roc auc if binary classification

            if self.args.criterion == 'bce':
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | AUC_ROC: {metric:.4f}')
            elif self.args.criterion == 'rmse':
                print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            if val_loss < best_loss:
                early_stopping = 0
                best_loss = val_loss

                torch.save(self.model.state_dict(), f'models/downstream/{self.data}_best_model.pth')
                print('Saved!')

            else:
                early_stopping += 1

            if early_stopping == self.args.patience:
                break

    def pre_train_one_epoch(self):
        self.model.train()
        train_loss = 0

        features, labels = [], []
        for smiles, second in tqdm(self.train_iter):
            smiles, second = smiles.to(self.device), second.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(smiles)

            loss = self.criterion(output, second)
            loss.backward()

            nn.utils.clip_grad_value_(self.model.parameters(), self.args.clipping)
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(self.train_iter)

        return train_loss

    def pre_validate(self):
        self.model.eval()
        gts, preds = [], []

        with torch.no_grad():
            val_loss = 0

            for smiles, second in tqdm(self.valid_iter):
                smiles, second = smiles.to(self.device), second.to(self.device)

                output = self.model(smiles)

                loss = self.criterion(output, second)
                val_loss += loss.item()

                preds += output.detach().cpu().numpy().tolist()
                gts += second.detach().cpu().numpy().tolist()


            val_loss /= len(self.valid_iter)

            # metric
            if self.args.criterion == 'bce':
                metric = roc_auc_score(gts, preds)
            else:
                metric = 0

        return val_loss, metric

    def test(self):
        if self.args.task == 'inference':
            self.model = Classifier(self.args, encoder=self.model, num_classes=self.args.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(f'models/downstream/{self.data}_best_model.pth'))

        # test
        self.model.eval()
        gts, preds = [], []

        with torch.no_grad():
            for smiles, second in tqdm(self.test_iter):
                smiles, second = smiles.to(self.device), second.to(self.device)

                output = self.model(smiles)

                preds += output.detach().cpu().numpy().tolist()
                gts += second.detach().cpu().numpy().tolist()

        # metric
        if self.args.criterion == 'bce':
            metric = roc_auc_score(gts, preds)
            print(f'{self.data} AUC_ROC: {metric:.4f}')
        elif self.args.criterion == 'rmse':
            metric = mean_squared_error(gts, preds)**0.5
            print(f'{self.data} RMSE: {metric:.4f}')























