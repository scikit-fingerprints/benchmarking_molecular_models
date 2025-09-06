import os
import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from SmilesEnumerator import SmilesEnumerator
from rdkit import Chem


def load_smiles(data):
    """
    Load SMILES from dataset.
    """
    if data == 'pubchem':
        df = pd.read_csv('old/data/pubchem.csv')
        smiles = df.isosmiles
    elif data == 'pubchem_part':
        df = pd.read_csv('old/data/pubchem_part.csv')
        smiles = df.smiles
    elif data == 'manufacturing':
        df = pd.read_csv('old/data/pubchem_manufacturing.csv')
        smiles = df.isosmiles
    elif data == 'spectral':
        df = pd.read_csv('old/data/pubchem_spectral.csv')
        smiles = df.isosmiles
    elif data == 'chembl':
        df = pd.read_csv('old/data/chembl.csv')
        smiles = df.canonical_smiles
    elif data == 'chembl_part':
        df = pd.read_csv('old/data/chembl_part.csv')
        smiles = df.smiles
    elif data == 'smiles10m':
        df = pd.read_csv('old/data/smiles_10m.csv')
        smiles = df.smiles
    elif data == 'bace':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv')
        smiles = df.mol
    elif data == 'tox21':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz', compression='gzip')
        df = df.dropna(axis=0, how='any').reset_index(drop=True)  # drop nan values
        smiles = df.smiles
    elif data == 'qm8':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv')
        df = df.dropna(axis=0, how='any').reset_index(drop=True)  # drop nan values
        smiles = df.smiles
    elif data == 'qm7':
        df = pd.read_csv('data/prediction/qm7.csv')
        df = df.dropna(axis=0, how='any').reset_index(drop=True)  # drop nan values
        smiles = df.smiles

    return smiles


def load_lists_from_url(data):
    """
    Load SMILES and labels from Moleculenet website.
    """
    if data == 'bbbp':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv')
        smiles, labels = df.smiles, df.p_np
    elif data == 'clintox':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz', compression='gzip')
        smiles = df.smiles
        labels = df.drop(['smiles'], axis=1)
    elif data == 'hiv':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv')
        smiles, labels = df.smiles, df.HIV_active
    elif data == 'sider':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz', compression='gzip')
        smiles = df.smiles
        labels = df.drop(['smiles'], axis=1)    # (1427, 27)
    elif data == 'esol':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv')
        smiles = df.smiles
        labels = df['ESOL predicted log solubility in mols per litre']
    elif data == 'freesolv':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv')
        smiles = df.smiles
        labels = df.calc
    elif data == 'lipophilicity':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv')
        smiles, labels = df.smiles, df['exp']
    elif data == 'tox21':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz', compression='gzip')
        df = df.dropna(axis=0, how='any').reset_index(drop=True)   # drop nan values
        smiles = df.smiles
        labels = df.drop(['mol_id', 'smiles'], axis=1)  # 12 cols
    elif data == 'bace':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv')
        smiles, labels = df.mol, df.Class
    elif data == 'tox21':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz', compression='gzip')
        df = df.dropna(axis=0, how='any').reset_index(drop=True)  # drop nan values
        smiles = df.smiles
        labels = df.drop(['mol_id', 'smiles'], axis=1)  # 12 cols
    elif data == 'qm8':
        df = pd.read_csv('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv')
        df = df.dropna(axis=0, how='any').reset_index(drop=True)  # drop nan values
        smiles = df.smiles
        labels = df.drop(['smiles', 'E2-PBE0.1', 'E1-PBE0.1', 'f1-PBE0.1', 'f2-PBE0.1'], axis=1)  # 12 tasks

    return smiles, labels


def csv_to_txt(smiles, data):
    my_str = ''
    for i in range(len(smiles)):
        my_str += str(smiles[i]) + '\n'

    save_path = 'data/' + str(data) + '.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(my_str)


def tokenize_enumerated_smiles(args):
    """
    Enumerate smiles and save tokenize smiles
    Input: list of smiles
    Output: [smiles, enumerated smiles] tensors
    """
    # load data
    smiles = load_smiles(args.data)

    # check validity
    valid_smiles = []
    cnt = 0
    for i in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[i])
        if mol:
            valid_smiles.append(smiles[i])
            cnt += 1
        if cnt == 10000000:
            break

    print('Valid smiles length: ', len(valid_smiles))

    # enumerate smiles
    sme = SmilesEnumerator()
    smiles_enumerated = []
    print('Starting enumeration...')
    for i in tqdm(range(len(valid_smiles))):
        smiles_enumerated.append(sme.randomize_smiles(valid_smiles[i]))

    # save a list of smiles to txt format
    txt_path = 'data/' + str(args.data) + '.txt'
    if os.path.exists(txt_path):
        pass
    else:
        csv_to_txt(valid_smiles, args.data)

    txt_path = 'data/' + str(args.data) + '.txt'

    # train tokenizer
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(special_tokens=['[PAD]', '[UNK]'], vocab_size=args.dic_size, min_frequency=args.min_frequency)
    tokenizer.enable_padding(pad_id=0, pad_token='[PAD]', length=args.max_len)
    tokenizer.enable_truncation(max_length=args.max_len)
    tokenizer.train([txt_path], trainer)

    # save tokenizer
    os.makedirs('data/tokenizer', exist_ok=True)
    tokenizer_path = 'data/tokenizer/' + str(args.data) + '_tokenizer.json'
    tokenizer.save(tokenizer_path)
    print('Saved the tokenizer!')

    # tokenize and check length
    tokenized = []
    tokenized2 = []

    print('Starting tokenization...')
    for i in range(len(valid_smiles)):
        output = tokenizer.encode(valid_smiles[i])
        output2 = tokenizer.encode(smiles_enumerated[i])

        tokenized.append(output.ids)
        tokenized2.append(output2.ids)

    print('Final data length: ', len(tokenized))

    # change to tensor
    tokenized = torch.LongTensor(tokenized)
    tokenized2 = torch.LongTensor(tokenized2)

    # save file
    os.makedirs('data/embedding', exist_ok=True)
    path = 'data/embedding/' + str(args.data) + '.pth'
    torch.save([tokenized, tokenized2], path)   # shape = [[dataset_len, max_len], [dataset_len, max_len]]
    print('Save the smiles-enumerated smiles tensors!')
    

def tokenize_smiles_labels(args, data, split, num_classes=1):
    """
    Tokenize smiles and labels for downstream task.
    Input: smiles, labels
    Output: [smiles, labels] tensors, list of valid idx
    """
    # load data
    smiles, labels = load_lists_from_url(data)

    # load tokenizer
    tokenizer_path = 'data/tokenizer/pubchem_part_tokenizer.json'
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # tokenize and check length
    tokenized = []
    idx = []
    print('Starting tokenization...')
    for i in tqdm(range(len(smiles))):
        output = tokenizer.encode(smiles[i])
        tokenized.append(output.ids)
        if len(output.ids) <= args.max_len:
            idx.append(int(i))

    # check validity
    if split == 'scaffold':
        print('Checking validity for scaffold split.')
        val_idx = []
        for i in range(len(smiles)):
            if Chem.MolFromSmiles(smiles[i]):
                val_idx.append(int(i))

        idx = list(set(idx).intersection(val_idx))
        idx_path = 'data/prediction/' + str(data) + '_idx'
        np.save(idx_path, np.array(idx))

    # save new lists
    tokenized_list = []
    labels_list = []
    for i in idx:
        tokenized_list.append(tokenized[i])
        if num_classes > 1:
            labels_list.append(labels.iloc[i])
        else:
            labels_list.append([labels[i]])

    # change to tensor
    tokenized_list = torch.LongTensor(tokenized_list)
    labels_list = torch.FloatTensor(labels_list)

    print('SMILES length: ', len(tokenized_list))
    print('Labels length: ', len(labels_list))

    # save
    os.makedirs('data/prediction', exist_ok=True)
    path = 'data/prediction/' + str(data) + '.pth'
    torch.save([tokenized_list, labels_list], path)
    print('Saved the smiles-labels tensors!')






