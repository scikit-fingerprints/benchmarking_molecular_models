import os
import numpy as np
import pandas as pd
import random
import torch
import pprint
import warnings
from trainer import Trainer
from tokenization import tokenize_enumerated_smiles, tokenize_smiles_labels
from config import get_Config

warnings.filterwarnings('ignore')

def set_seed(args):
    np.seterr(all="ignore")
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize_all(args):
    # tokenize pretrain dataset
    if os.path.exists('data/embedding/pubchem_part.pth'):
        pass
    else:
        tokenize_enumerated_smiles(args)

    # tokenize downstream datasets
    datasets = ['tox21', 'bbbp', 'clintox', 'hiv', 'bace', 'sider', 'esol', 'freesolv', 'lipophilicity', 'bace', 'sider', 'qm8', 'qm7']
    for data in datasets:
        path = 'data/prediction/' + data + '.pth'
        if os.path.exists(path):
            pass
        else:
            tokenize_smiles_labels(args, data, split=args.split, num_classes=args.num_classes)


def main(args):
    print('<---------------- Training params ---------------->')
    pprint.pprint(args)

    # Random seed
    set_seed(args)

    # tokenize
    tokenize_all(args)

    # train
    if args.task == 'pretraining':
        trainer = Trainer(args, data='pubchem_part')
        trainer.train()

    elif args.task == 'downstream':
        trainer = Trainer(args, data=args.data)
        trainer.pre_train()
        trainer.test()
    elif args.task == 'inference':
        trainer = Trainer(args, data=args.data)
        trainer.test()


if __name__ == '__main__':
    args = get_Config()
    main(args)








