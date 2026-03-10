import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, feature_dim,**kwargs):
        super(ChannelAttention, self).__init__()
        self.mask = torch.nn.Parameter(torch.zeros((1, feature_dim, 1, 1)))
        self.eps = 1e-8

    def forward(self, reverse=False, detach=False):
        #soft mask to hard mask[0,1]
        mask = torch.cat([self.mask + self.eps,torch.zeros_like(self.mask)],dim=0)
        if self.training:
            hard_mask = F.gumbel_softmax(mask, tau=1 ,hard=True, dim=0)
        else:
            hard_mask = F.softmax(mask, dim=0)
            hard_mask = (hard_mask >= 0.5).float()
        if not reverse:
            hard_mask = hard_mask[0]
        else:
            hard_mask = hard_mask[1]
        hard_mask =  hard_mask.unsqueeze(0)
        if detach: hard_mask = hard_mask.detach()
        return hard_mask

class ChannelDrop(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(ChannelDrop, self).__init__()
        self.omega = nn.Parameter(torch.zeros((1,feature_dim,1,1)))
        self.eps = 1e-8
        self.u = torch.rand_like(self.omega).cuda()

    @property
    def p(self):
        return self.omega.sigmoid()

    def forward(self, reverse=False, meta=False):
        if self.training:
            p = self.p
            if not meta:
                self.u = torch.cat([torch.rand_like(p) for i in range(x.shape[0])])
                mask = (self.u<p.repeat(x.shape[0],1,1,1)).float()
            else:
                if not reverse:
                    self.u = torch.rand_like(p)
                    mask = (self.u<p).float()
                else:
                    mask = (self.u > (1-p).float())
        else:
            mask = self.p
        return mask

class ChannelAttentionFlex(nn.Module):
    def __init__(self, feature_dim,**kwargs):
        super(ChannelAttentionFlex, self).__init__()
        self.mask = torch.nn.Parameter(torch.ones((1, feature_dim, 1, 1)))
        self.eps = 1e-8

    def forward(self,**kwargs):

        hard_mask = torch.nn.functional.softplus(self.mask)
        return  hard_mask

class ChannelAttentionSoft(nn.Module):
    def __init__(self, feature_dim,*args,**kwargs):
        super(ChannelAttentionSoft, self).__init__()
        self.mask = torch.nn.Parameter(torch.zeros((1, feature_dim, 1, 1)))
        self.eps = 1e-8

    def forward(self, *args,**kwargs):

        mask = torch.nn.functional.sigmoid(self.mask)
        mask = mask
        return mask

class ChannelAttentionLinear(nn.Module):
    def __init__(self, feature_dim, *args, **kwargs):
        super(ChannelAttentionLinear, self).__init__()
        self.net = torch.nn.Conv2d(feature_dim,feature_dim,(1,1),(1,1),0)
        self.eps = 1e-8

    def forward(self, x, reverse=False, detach=False, *args,**kwargs):
        x = x.mean(dim=(0,-1,-2),keepdim=True)
        x = self.net(x)

        return x

class ChannelAttentionNonLinear(nn.Module):
    def __init__(self, feature_dim, *args, **kwargs):
        super(ChannelAttentionNonLinear, self).__init__()
        self.net = nn.Sequential(
            torch.nn.Conv2d(feature_dim,int(feature_dim/2),(1,1),(1,1),0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(int(feature_dim/2),feature_dim,(1,1),(1,1),0)
        )
        self.eps = 1e-8

        for m in self.net.modules():
            if isinstance(m,torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0,std=0.01)

    def forward(self, x, *args, **kwargs):
        x = x.mean(dim=(-1,-2),keepdim=True)
        x = x.var(dim=0,keepdim=True)
        x = x/x.max()
        x = self.net(x)
        # soft mask to hard mask[0,1]
        self.mask = x
        mask = torch.cat([x + self.eps, torch.zeros_like(x)], dim=0)

        if self.training:
            hard_mask = F.gumbel_softmax(mask, tau=1, hard=True, dim=0)
        else:
            hard_mask = F.softmax(mask, dim=0)
            hard_mask = (hard_mask >= 0.5).float()
        hard_mask = hard_mask[0]
        hard_mask = hard_mask.unsqueeze(0)
        return hard_mask

class ChannelAttentionNonLinearSoft(nn.Module):
    def __init__(self, feature_dim, *args, **kwargs):
        super(ChannelAttentionNonLinearSoft, self).__init__()
        self.net = nn.Sequential(
            torch.nn.Conv2d(feature_dim,int(feature_dim/2),(1,1),(1,1),0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(int(feature_dim/2),feature_dim,(1,1),(1,1),0)
        )
        self.eps = 1e-8

        for m in self.net.modules():
            if isinstance(m,torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0,std=0.01)

    def forward(self, x, *args, **kwargs):
        x = x.mean(dim=(0,-1,-2),keepdim=True)
        x = self.net(x)
        # soft mask to hard mask[0,1]
        self.mask = x.sigmoid()
        return self.mask

class ChannelAttentionRandom(nn.Module):
    def __init__(self, feature_dim,**kwargs):
        super(ChannelAttentionRandom, self).__init__()
        self.mask = torch.nn.Parameter(torch.randn((1, feature_dim, 1, 1)))
        self.eps = 1e-8

    def forward(self, reverse=False):
        with torch.no_grad():
            #soft mask to hard mask[0,1]
            mask = torch.cat([self.mask + self.eps,torch.zeros_like(self.mask)],dim=0)
            hard_mask = F.softmax(mask, dim=0)
            hard_mask = (hard_mask >= 0.5).float()
            if not reverse:
                hard_mask = hard_mask[0]
            else:
                hard_mask = hard_mask[1]
            hard_mask =  hard_mask.unsqueeze(0)

        return hard_mask

class LearnableMaskLayer(nn.Module):
    def __init__(self, feature_dim):
        super(LearnableMaskLayer, self).__init__()
        self.mask = torch.nn.Parameter(1e-3 * torch.randn((2, feature_dim, 1, 1)))

    def forward(self, domain_flag):
        # soft mask to hard mask[0,1]
        if self.training:
            hard_mask = F.gumbel_softmax(self.mask, hard=True, dim=0)
        else:
            hard_mask = F.softmax(self.mask, dim=0)
            hard_mask = (hard_mask > 0.5).float()
            # print('S:', torch.sum(hard_mask[0]))
        # print('A:', torch.sum(hard_mask[1]))
        if (domain_flag == 'S'):
            hard_mask = hard_mask[0]
        elif (domain_flag == 'A'):
            hard_mask = hard_mask[1]

        hard_mask = hard_mask.unsqueeze(0)
        return hard_mask

class DASD(nn.Linear):
    def __init__(self, indim, outdim=2, drop_rate=0.5, *args, **kwargs):
        super(DASD, self).__init__(in_features=indim, out_features=outdim, bias=False)
        self.dim = indim
        self.drop_rate = 1-drop_rate
        self.p = torch.ones(indim) / indim

    def forward(self, input, data_flag='S', reverse=False, *args, **kwargs):
        x = input.mean(dim=(-1, -2))
        if self.training:
            if data_flag == 'S':
                logits = self.weight[0].unsqueeze(0) * x
            else:
                logits = self.weight[1].unsqueeze(0) * x

            scores = torch.exp(logits * 1)
            self.p = (scores / scores.sum(dim=1, keepdim=True)).mean(dim=0)
            r = torch.rand_like(x)
            key = r ** (1 / scores)
            importance_order = key.argsort(dim=1, descending=not reverse)

            mask = torch.ones_like(x)
            if self.drop_rate > 0:
                mask_index = importance_order[:, :int(self.dim * self.drop_rate)]
                mask = mask.scatter(dim=1, index=mask_index, value=0)
        else:
            mask = (1 - self.drop_rate) * torch.ones_like(x)

        output = input * mask.unsqueeze(-1).unsqueeze(-1)
        return output

if __name__ == '__main__':
    myLearnableMaskLayer = ChannelAttention(feature_dim = 8)
    x = torch.randn(2,8,5,5)
    out_x = myLearnableMaskLayer()
    print(out_x.shape)
    print(out_x.sum((2,3)))
    out_x.sum().backward()
    print(myLearnableMaskLayer.mask.grad)
