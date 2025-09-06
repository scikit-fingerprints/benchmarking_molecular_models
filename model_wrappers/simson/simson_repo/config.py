import argparse


def get_Config():
    parser = argparse.ArgumentParser()

    # training setting
    parser.add_argument('--task', type=str, default='pretraining', help='pretraining / downstream / inference')
    parser.add_argument('--data', type=str, default='pubchem_part')
    parser.add_argument('--pre_exp_num', type=int, default=None, help='Experiment number to load pretrained model')
    parser.add_argument('--epoch_num', type=int, default=3, help='Epoch number to load pretrained model')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--criterion', type=str, default='contrastive')
    parser.add_argument('--epoch_loss', type=float, default=0, help='for downstream experiment log')

    # tokenization parameters
    parser.add_argument('--dic_size', type=int, default=300, help='Number of BPE vocabularies')
    parser.add_argument('--min_frequency', type=int, default=2, help='for BPE tokenization')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--split', type=str, default='random', help='scaffold split or random split')

    # transformer parameters
    parser.add_argument('--d_model', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)

    # training parameters
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW / Adam / SGD')
    parser.add_argument('--scheduler', type=str, default='cos', help='MultiStep / CosineAnnealing / OneCycle')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='CosineAnnealing scheduler parameter')
    parser.add_argument('--warm_epoch', type=int, default=5, help='warmup scheduler')
    parser.add_argument('--tmax', type=int, default=145, help='CosineAnnealing scheduler parameter')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='OneCycle scheduler parameter')
    parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
    parser.add_argument('--temperature', type=float, default=0.2, help='Contrastive loss temperature')

    # hardware setting
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    return args

