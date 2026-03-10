import os
import torch
import numpy as np
import torch.nn as nn
from methods import gnn
from methods.gnn import GNN_nl
from torch.nn import functional as F
from methods import backbone as backbone
from utils import yellow_text, green_text, ensure_path, create_txt
from methods.meta_template_metaChannelAttention import MetaTemplate
from torchvision.utils import save_image
from methods.backbone_ffilter import FreqFilter
from matplotlib import cm
import cv2
from PIL import Image

class_categories = {}
class_categories['source'] = 64
class_categories['cub'] = 100
class_categories['cars'] = 98
class_categories['places'] = 183
class_categories['plantae'] = 100
class_categories['sketch'] = 173
class_categories['infograph'] = 173
class_categories['eurosat'] = 5
class_categories['traffic'] = 26
EPS = 0.00001

class SKL(torch.nn.Module):
    def __init__(self, temperature=5.0):
        super(SKL, self).__init__()
        self.temp = temperature
        self.KL = torch.nn.KLDivLoss()

    def forward(self, score_t, score_s):
        log_dist_t = F.log_softmax(score_t / self.temp, dim=1)
        dist_t = F.softmax(score_t / self.temp, dim=1)
        log_dist_s = F.log_softmax(score_s / self.temp, dim=1)
        dist_s = F.softmax(score_s / self.temp, dim=1)
        return 0.5 * (self.KL(log_dist_t, dist_s) + self.KL(log_dist_s, dist_t))

@torch.no_grad()
def swap_low_freq_mutual(S, A, gamma=0.2):
    """
    对两个 batch 的图像做低频互换，保留各自高频，互换低频。

    Args:
        S: [B, C, H, W] tensor
        A: [B, C, H, W] tensor
        radius: int, 控制低频掩码的半径
    Returns:
        S_recon: S 的高频 + A 的低频
        A_recon: A 的高频 + S 的低频
    """
    S = S.cuda()
    A = A.cuda()

    N, B, C, H, W = S.size()
    S = S.view(-1, *S.size()[2:])
    A = A.view(-1, *A.size()[2:])
    assert S.shape == A.shape, "S and A must have the same shape"

    # 1. 频域变换 + 中心化
    fft_S = torch.fft.fftshift(torch.fft.fft2(S), dim=(-2, -1))
    fft_A = torch.fft.fftshift(torch.fft.fft2(A), dim=(-2, -1))

    # 2. 构建低频掩码
    radius =  gamma * np.random.beta(1.0,1.0) * 224
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = torch.sqrt((X - W // 2) ** 2 + (Y - H // 2) ** 2).to(S.device)
    low_mask = (dist < radius).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # 3. 分离高频 & 低频
    S_low, A_low = fft_S * low_mask, fft_A * low_mask
    S_high, A_high = fft_S * (1 - low_mask), fft_A * (1 - low_mask)

    # 4. 互换低频 + 保留高频
    fft_S_new = S_high + A_low
    fft_A_new = A_high + S_low

    # 5. IFFT + shift
    S_recon = torch.fft.ifft2(torch.fft.ifftshift(fft_S_new, dim=(-2, -1))).real
    A_recon = torch.fft.ifft2(torch.fft.ifftshift(fft_A_new, dim=(-2, -1))).real
    S_recon = S_recon.view(N, B, C, H, W)
    A_recon = A_recon.view(N, B, C, H, W)

    return S_recon, A_recon

class GnnNetStudent(MetaTemplate):
    maml = False
    def __init__(self, model_func, args, n_way, n_support, tf_path=None, target_set='None'):
        super(GnnNetStudent, self).__init__(model_func, args, n_way, n_support, tf_path=tf_path)
        if self.maml:
            gnn.Gconv.maml = True
            gnn.Wcompute.maml = True
        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # metric function
        self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128),
                                nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(
            backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
        self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)

        self.method = 'GnnNet'
        # define global fc classifiers
        self.classifier_source = nn.Linear(self.feat_dim,
                                           class_categories['source']) if not self.maml else backbone.Linear_fw(
            self.feat_dim, class_categories['source'])
        self.classifier_target = nn.Linear(self.feat_dim,
                                           class_categories[target_set]) if not self.maml else backbone.Linear_fw(
            self.feat_dim, class_categories[target_set])

        # fix label for training the metric function   1*nw(1 + ns)*nw
        support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
        support_label = torch.zeros(self.n_way * self.n_support, self.n_way).scatter(1, support_label, 1).view(
            self.n_way, self.n_support, self.n_way)

        support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)

        self.support_label = support_label.view(1, -1, self.n_way)
        self.skl = SKL()

        param = self.split_model_parameters()
        self.model_param = param

        self.freq_filter1  = FreqFilter(64,56,56)
        self.freq_filter2  = FreqFilter(128,28,28)
        self.freq_filter3  = FreqFilter(256,14,14)
        self.freq_filter4  = FreqFilter(512,7,7)

        self.feature.trunk[6].drop_prob = self.args.drop_prob
        self.feature.trunk[7].drop_prob = self.args.drop_prob

        self.feature.trunk[6].FE = self.args.FE
        self.feature.trunk[7].FE = self.args.FE

    def cuda(self):
        self.feature.cuda()
        self.fc.cuda()
        self.gnn.cuda()
        self.support_label = self.support_label.cuda()
        self.classifier_source.cuda()
        self.classifier_target.cuda()
        self.freq_filter1.cuda()
        self.freq_filter2.cuda()
        self.freq_filter3.cuda()
        self.freq_filter4.cuda()
        return self

    def set_statues_of_modules(self, flag):
        if (flag == 'eval'):
            self.feature.eval()
            self.fc.eval()
            self.gnn.eval()
            self.classifier_source.eval()
            self.classifier_target.eval()
        elif (flag == 'train'):
            self.feature.train()
            self.fc.train()
            self.gnn.train()
            self.classifier_source.train()
            self.classifier_target.train()
        return

    def split_model_parameters(self, return_name=False):
        model_params = []
        model_params_name = []

        for n, p in self.named_parameters():
            n = n.split('.')
            model_params.append(p)
            model_params_name.append(n)
        if not return_name:
            return model_params
        else:
            return model_params,model_params_name

    def standard_path(self, x,domain=None,stop_grad=False, **kwargs):
        self.freq_filter1.stop_grad = stop_grad
        self.freq_filter2.stop_grad = stop_grad
        self.freq_filter3.stop_grad = stop_grad
        self.freq_filter4.stop_grad = stop_grad

        # standard path
        self.freq_filter1.domain=domain
        self.freq_filter2.domain=domain
        self.freq_filter3.domain=domain
        self.freq_filter4.domain=domain

        if self.args.FF:
            self.x_fea_block1 = self.feature.forward_block1(x)
            self.x_fea_block1 = self.freq_filter1(self.x_fea_block1)
            self.x_fea_block2 = self.feature.forward_block2(self.x_fea_block1)
            self.x_fea_block2 = self.freq_filter2(self.x_fea_block2)
            self.x_fea_block3 = self.feature.forward_block3(self.x_fea_block2)
            self.x_fea_block3 = self.freq_filter3(self.x_fea_block3)
            self.x_fea_block4 = self.feature.forward_block4(self.x_fea_block3)
            self.x_fea_block4 = self.freq_filter4(self.x_fea_block4)
        else:
            self.x_fea_block1 = self.feature.forward_block1(x)
            self.x_fea_block2 = self.feature.forward_block2(self.x_fea_block1)
            self.x_fea_block3 = self.feature.forward_block3(self.x_fea_block2)
            self.x_fea_block4 = self.feature.forward_block4(self.x_fea_block3)
        self.x_fea = self.feature.forward_rest(self.x_fea_block4)

        return self.x_fea

    def forward_gnn(self, zs):
        # gnn inp: n_q * n_way(n_s + 1) * f
        # print(zs.shape)
        nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
        scores = self.gnn(nodes)
        # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
        scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0,
                                                                                                         2).contiguous().view(
            -1, self.n_way)
        return scores

    def predict(self, fea):
        z = self.fc(fea)
        z = z.view(self.n_way, -1, z.size(1))
        # print(z.shape)
        z_stack = [
            torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1,
                                                                                                            z.size(2))
            for i in range(self.n_query)]
        assert (z_stack[0].size(1) == self.n_way * (self.n_support + 1))
        # print('z_stack:', 'len:', len(z_stack), 'z_stack[0]:', z_stack[0].size())
        scores = self.forward_gnn(z_stack)
        return scores

    def feat_predict(self, x):
        x = x.cuda()
        # reshape the feature tensor: n_way * n_s + 15 * f
        assert (x.size(1) == self.n_support + 15)
        x = x.view(-1, *x.size()[2:])
        return self.predict(x)

    def get_classification_scores(self, z, classifier):
        z_norm = torch.norm(z, p=2, dim=1).unsqueeze(1).expand_as(z)
        z_normalized = z.div(z_norm + EPS)
        L_norm = torch.norm(classifier.weight.data, p=2, dim=1).unsqueeze(1).expand_as(classifier.weight.data)
        classifier.weight.data = classifier.weight.data.div(L_norm + EPS)
        cos_dist = classifier(z_normalized)
        cos_fac = 1.0
        scores = cos_fac * cos_dist
        return scores

    def cls_loss(self,x_fea,label,domain='S'):
        if (domain == 'S'):
            cls_scores = self.get_classification_scores(x_fea, self.classifier_source)
            y_cls = label.view(cls_scores.size()[0]).cuda()
            cls_loss = self.loss_fn(cls_scores, y_cls)
        elif (domain == 'A'):
            cls_scores = self.get_classification_scores(x_fea, self.classifier_target)
            y_cls = label.view(cls_scores.size()[0]).cuda()
            cls_loss = self.loss_fn(cls_scores, y_cls)
        else:
            cls_loss=0
        return cls_loss

    def fsl_loss(self,x_fea):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        # todo fsl loss
        fsl_scores = self.predict(x_fea)
        fsl_loss = self.loss_fn(fsl_scores, y_query)
        return fsl_scores, fsl_loss

    def forward_loss(self, x, y_cls, data_flag, standard_path=True):
        x = x.cuda()
        x = x.view(-1, *x.size()[2:])

        if self.training:
            if (data_flag == 'S' and self.args.ffilt_S) or (data_flag == 'A' and self.args.ffilt_A):
                x_fea = self.standard_path(x, domain=data_flag)
            elif self.args.share_ffilt:
                x_fea = self.standard_path(x, domain='A')
            elif data_flag == 'S' and self.args.A_ffilt_S:
                x_fea = self.standard_path(x, domain='A', stop_grad=True)
            elif data_flag == 'A' and self.args.S_ffilt_A:
                x_fea = self.standard_path(x, domain='S', stop_grad=True)
            else:
                x_fea = self.standard_path(x, domain=None)
        else:
            if self.args.S_ffilt_A:
                x_fea = self.standard_path(x, domain='S')
            else:
                x_fea = self.standard_path(x,domain='A')

        fsl_scores, fsl_loss = self.fsl_loss(x_fea)

        # todo cls loss
        cls_loss = self.cls_loss(x_fea,y_cls,data_flag)

        return fsl_scores, fsl_loss, cls_loss

    def FSL_Loss(self,x):
        x = x.cuda()
        x = x.view(-1, *x.size()[2:])
        x_fea = self.standard_path(x, domain=None)
        fsl_scores, fsl_loss = self.fsl_loss(x_fea)
        return fsl_scores, fsl_loss

    def train_loop(self, epoch, S_train_loader, A_train_loader, optimizer, total_it):
        def zero_grads():
            optimizer.zero_grad()

        print_freq = len(S_train_loader) // 10
        avg_loss = 0

        for ((i, (S_x, S_y_global)), (i, (A_x, A_y_global))) in zip(enumerate(S_train_loader), enumerate(A_train_loader)):
            self.n_query = S_x.size(1) - self.n_support
            if self.change_way:
                self.n_way = S_x.size(0)

            # DSG: Forward Student wth Mask
            S_scores, S_loss_fsl, S_loss_cls = self.forward_loss(S_x, S_y_global, data_flag='S')

            A_scores, A_loss_fsl, A_loss_cls = self.forward_loss(A_x, A_y_global, data_flag='A')

            if self.args.FM:
                S_x, A_x = swap_low_freq_mutual(S_x, A_x, self.args.gamma)
                S_scores2, S_loss_fsl2 = self.FSL_Loss(S_x)
                A_scores2, A_loss_fsl2 = self.FSL_Loss(A_x)
                S_loss2 = S_loss_fsl2
                A_loss2 = A_loss_fsl2
            else:
                S_loss2 = A_loss2 = 0

            # STD: loss
            S_loss_base = S_loss_cls + S_loss_fsl
            A_loss_base = A_loss_fsl + A_loss_cls
            loss = 0.2 * S_loss_base + A_loss_base + 0.2 * S_loss2 + A_loss2
            zero_grads()
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if (total_it + 1) % 1 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar(self.method + '/total_loss', loss.item(), total_it + 1)
                self.tf_writer.add_scalar(self.method + '/src_fsl_loss', S_loss_fsl.item(), total_it + 1)
                self.tf_writer.add_scalar(self.method + '/tar_fsl_loss', A_loss_fsl.item(), total_it + 1)

            if (i + 1) % print_freq == 0:
                print(
                    yellow_text(f'Epoch {epoch} ') + '| Batch {:d}/{:d} | Loss {:f}'.format(i + 1, len(S_train_loader),
                                                                                            avg_loss / float(i + 1)),
                    end='\n' if (i + 1) == len(S_train_loader) else '\r')

            if (total_it) % 20 == 0 and self.tf_writer is not None:
                try:
                    pass
                except:
                    pass

            if (total_it) % 100 == 1 and self.tf_writer is not None:
                save_dir = os.path.join(self.tf_path, 'results')
                ensure_path(save_dir)
                layers = [1, 2, 3, 4]
                for i in layers:
                    freq_filter = getattr(self,f'freq_filter{i}')
                    name_prefix = f"layer{i}"

                    # 取半参数
                    half_S = freq_filter.filt_half_S.detach()
                    half_A = freq_filter.filt_half_A.detach()

                    # 构造对称滤波器
                    full_S = freq_filter._build_symmetric_filter(half_S)
                    full_A = freq_filter._build_symmetric_filter(half_A)

                    # 取所有 channel 的平均
                    avg_S = full_S.mean(dim=1)[0]  # shape (H, W)
                    avg_A = full_A.mean(dim=1)[0]

                    # 归一化
                    def normalize(x):
                        return (x - x.min()) / (x.max() - x.min() + 1e-8)

                    avg_S_norm = normalize(avg_S).unsqueeze(0)  # (1, H, W)
                    avg_A_norm = normalize(avg_A).unsqueeze(0)

                    # 保存图像
                    save_image(avg_S_norm, os.path.join(save_dir, f'{total_it}_{name_prefix}_F_S_avg.png'))
                    save_image(avg_A_norm, os.path.join(save_dir, f'{total_it}_{name_prefix}_F_A_avg.png'))

            total_it += 1
        return total_it

    def test_loop(self, test_loader, record=None, prefix=''):
        loss = 0
        count = 0
        acc_all = []
        acc_all_mask = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, correct_mask, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
            acc_all_mask.append(correct_mask / count_this * 100)
            loss += loss_this
            count += count_this

        acc_all_mask = np.asarray(acc_all_mask)
        acc_mask_mean = np.mean(acc_all_mask)
        acc_mask_std = np.std(acc_all_mask)
        print(f'--- {prefix} %d Loss = %.6f ---' % (iter_num, loss / count))
        print(f'--- {prefix} %d Mask Test Acc = %4.2f%% +- %4.2f%% ---' % (
            iter_num, acc_mask_mean, 1.96 * acc_mask_std / np.sqrt(iter_num)))
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(f'--- {prefix} %d Test Acc = %4.2f%% +- %4.2f%% ---' % (
            iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        return acc_mean, acc_std, acc_mask_mean, acc_mask_std