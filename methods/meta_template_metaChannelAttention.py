
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
from methods import backbone

class MetaTemplate(nn.Module):
    maml=False

    def __init__(self, model_func, args, n_way, n_support, flatten=True, leakyrelu=False, tf_path=None, change_way=True):
        super(MetaTemplate, self).__init__()
        if self.maml:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.ResNet.maml = True
        self.args = args
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # (change depends on input)
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None
        self.tf_path = tf_path
        self.feature = model_func(flatten=flatten, leakyrelu=leakyrelu,outdim=args.outdim,se_layers=args.se_layers)
        self.feat_dim = self.feature.final_feat_dim

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        return z_support, z_query

    def correct(self, x):
        out = self.forward_loss(x,None, None, standard_path=True)
        scores, loss = out[:2]
        scores_mask = self.forward_loss(x, None, None, standard_path=False)[0]

        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)

        topk_scores, topk_labels = scores_mask.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct2 = np.sum(topk_ind[:, 0] == y_query)

        return float(top1_correct),float(top1_correct2), len(y_query), loss.item() * len(y_query)

    def test_loop(self, test_loader, record=None):
        loss = 0.
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
            loss += loss_this
            count += count_this

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('--- %d Loss = %.6f ---' % (iter_num, loss / count))
        print('--- %d Test Acc = %4.2f%% +- %4.2f%% ---' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean

