import numpy as np
from itertools import compress
from collections import defaultdict
import torch
import torch.nn as nn
import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tokenization import load_lists_from_url


class GenerateDataset(Dataset):
    def __init__(self, args, smiles, second, phase='train', train_idx=None, valid_idx=None, test_idx=None):
        if args.task == 'pretraining':
            smiles_train, smiles_val, second_train, second_val = train_test_split(smiles, second, test_size=0.2,
                                                                                  shuffle=True, random_state=args.seed)
        else:   # downstream
            if train_idx is not None:   # scaffold split index is given
                smiles_train, smiles_val, smiles_test = smiles[train_idx], smiles[valid_idx], smiles[test_idx]
                second_train, second_val, second_test = second[train_idx], second[valid_idx], second[test_idx]
            else:   # random split
                smiles_train, smiles_tv, second_train, second_tv = train_test_split(smiles, second, test_size=0.2,
                                                                                    shuffle=True, random_state=args.seed)
                smiles_val, smiles_test, second_val, second_test = train_test_split(smiles_tv, second_tv,
                                                                                      test_size=0.5, shuffle=False,
                                                                                      random_state=args.seed)

        if phase == 'train':
            self.smiles, self.second = smiles_train, second_train
        elif phase == 'valid' or 'val':
            self.smiles, self.second = smiles_val, second_val
        elif phase == 'test':
            self.smiles, self.second = smiles_test, second_test

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        second = self.second[idx]
        return smiles, second


class ScaffoldSplitter(nn.Module): # code referred from GraphMVP
    def __init__(self, data, seed, train_frac=0.8, val_frac=0.1, test_frac=0.1, include_chirality=True):
        self.data = data
        self.seed = seed
        self.include_chirality = include_chirality
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

    def generate_scaffold(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=self.include_chirality)
        return scaffold

    def scaffold_split(self):
        smiles, labels = load_lists_from_url(self.data)
        non_null = np.ones(len(smiles)) == 0

        if self.data == 'tox21' or self.data == 'sider' or self.data == 'clintox':
            for i in range(len(smiles)):
                if Chem.MolFromSmiles(smiles[i]) and labels.loc[i].isnull().sum() == 0:     # valid data
                    non_null[i] = 1
        else:
            for i in range(len(smiles)):
                if Chem.MolFromSmiles(smiles[i]):
                    non_null[i] = 1

        smiles_list = list(compress(enumerate(smiles), non_null))

        rng = np.random.RandomState(self.seed)

        scaffolds = defaultdict(list)
        for i, sms in smiles_list:
            scaffold = self.generate_scaffold(sms)
            scaffolds[scaffold].append(i)

        scaffold_sets = rng.permutation(list(scaffolds.values()))

        n_total_val = int(np.floor(self.val_frac * len(smiles_list)))
        n_total_test = int(np.floor(self.test_frac * len(smiles_list)))

        train_idx, val_idx, test_idx = [], [], []

        for scaffold_set in scaffold_sets:
            if len(val_idx) + len(scaffold_set) <= n_total_val:
                val_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        return train_idx, val_idx, test_idx












