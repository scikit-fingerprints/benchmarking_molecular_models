import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def calc_similarity_batch(self, smiles, iupac):
        representations = torch.cat([smiles, iupac], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        # proj_1, proj_2: [B, emb_dim]
        batch_size = proj_1.shape[0]
        mask = (~torch.eye(batch_size*2, batch_size*2, dtype=bool)).float()
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = mask.to(similarity_matrix.device) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def Optimizer(args, model):
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return optimizer


def Scheduler(args, optimizer):
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=args.lr_factor,
                                                         verbose=False)
    elif args.scheduler == 'cos':
        tmax = args.tmax
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=args.min_lr,
                                                               verbose=False)
    elif args.scheduler == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, epochs=args.epochs,
                                                        steps_per_epoch=10)
    elif args.scheduler == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor,
                                                               patience=args.patience)
    elif args.scheduler == 'cos_warm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warm_epoch)

    return scheduler





