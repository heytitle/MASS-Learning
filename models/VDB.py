import copy

import torch

from torch.nn import functional as F

from models.VIB import VIB
from models.utils import MOG


class VDB(VIB):
    def __init__(self, *args, **kwargs):
        super(VDB, self).__init__(*args, **kwargs)

    def loss(self, input, output, target):
        mean, std = self.encode(output)
        if self.training:
            n_samples = self.train_var_dist_samples
        else:
            n_samples = self.test_var_dist_samples
        rep = self.sample_representation(mean, std, n_samples)
        rep = rep.reshape(-1, self.rep_dim)

        logits = self.decoder(rep)

        logits = logits.reshape(n_samples, -1, self.n_classes)
        max_logits, _ = torch.max(logits, dim=2, keepdim=True)
        logits = logits - max_logits

        sm = F.softmax(logits, dim=1)
        mean_sm = torch.mean(sm, 0)

        target_onehot = torch.zeros(target.shape[0], 10)
        target_onehot[range(target.shape[0]), target] = 1

        vdb_class_loss = torch.mean( # average across all samples in batch
            - torch.sum(torch.log(mean_sm) * target_onehot, 1)
        )

        kl_term = self.beta * self.vib_loss_kl_term(n_samples, rep, mean, std)

        loss = vdb_class_loss + kl_term
        if self.training and self.writer.global_step % self.writer.train_loss_plot_interval == 0:
            self.writer.add_scalar('Train Loss/VDB class_loss', vdb_class_loss, self.writer.global_step)
            self.writer.add_scalar('Train Loss/VIB KL term', kl_term, self.writer.global_step)
        return loss