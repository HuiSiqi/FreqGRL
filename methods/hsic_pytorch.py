"""
Hilbert Schmidt Information Criterion with a Gaussian kernel, based on the
following references
[1]: https://link.springer.com/chapter/10.1007/11564089_7
[2]: https://www.researchgate.net/publication/301818817_Kernel-based_Tests_for_Joint_Independence

"""
import torch
import time

def centering(M):
    """
    Calculate the centering matrix
    """
    n = M.shape[0]
    unit = torch.ones([n, n])
    identity = torch.eye(n)
    H = identity - unit/n
    return M@H

def gaussian_grammat(x, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    ||x_i - x_j||**2 = 2 sigma**2
    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(x.shape[0], 1)

    xxT = x@x.T
    xnorm = torch.diag(xxT) - xxT + (torch.diag(xxT) - xxT).T
    if sigma is None:
        mdist = torch.median(xnorm[xnorm!= 0])
        sigma = torch.sqrt(mdist*0.5)


   # --- If bandwidth is 0, add machine epsilon to it
    if sigma==0:
        eps = 7./3 - 4./3 - 1
        sigma += eps

    KX = - 0.5 * xnorm / sigma / sigma
    KX = torch.exp(KX)
    return KX

def dHSIC_calc(K_list):
    """
    Calculate the HSIC estimator in the general case d > 2, as in
    [2] Definition 2.6
    """
    if not isinstance(K_list, list):
        K_list = list(K_list)

    n_k = len(K_list)

    length = K_list[0].shape[0]
    term1 = 1.0
    term2 = 1.0
    term3 = 2.0/length

    for j in range(0, n_k):
        K_j = K_list[j]
        term1 = term1*K_j
        term2 = 1.0/length/length*term2*torch.sum(K_j)
        term3 = 1.0/length*term3*K_j.sum(dim=0)

    term1 = torch.sum(term1)
    term3 = torch.sum(term3)
    dHSIC = (1.0/length)**2*term1+term2-term3
    return dHSIC

def HSIC(x, y):
    """
    Calculate the HSIC estimator for d=2, as in [1] eq (9)
    """
    n = x.shape[0]
    return torch.trace(centering(gaussian_grammat(x))@centering(gaussian_grammat(y)))/n/n

def dHSIC(*argv):
    assert len(argv) > 1, "dHSIC requires at least two arguments"

    if len(argv) == 2:
        x, y = argv
        return HSIC(x, y)
    start = time.time()
    K_list = [gaussian_grammat(_arg) for _arg in argv]
    stop = time.time()
    print(f'cost:{stop - start}')
    dHSIC =  dHSIC_calc(K_list)
    return dHSIC

def batch_gaussian_grammat(x, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    ||x_i - x_j||**2 = 2 sigma**2
    """
    #x: cXn
    x = x.unsqueeze(-1)
    xxT = torch.bmm(x,x.transpose(-1,-2).contiguous())
    diag = torch.diagonal(xxT,dim1=-1,dim2=-2).unsqueeze(-1)
    xnorm = diag-xxT
    xnorm = xnorm + xnorm.transpose(-1,-2).contiguous()
    if sigma is None:
        # mdist = torch.stack([torch.median(xn[xn!=0]) for xn in xnorm])
        mdist = torch.median(xnorm.view(xnorm.shape[0],-1),dim=-1).values
        print(mdist)
        sigma = torch.sqrt(mdist*0.5)

   # --- If bandwidth is 0, add machine epsilon to it
    eps = 7. / 3 - 4. / 3 - 1
    sigma = sigma + (sigma==0)*eps

    sigma = sigma.unsqueeze(-1).unsqueeze(-1)
    sigma = sigma.to(xnorm.device)
    KX = - 0.5 * xnorm / sigma / sigma
    KX = torch.exp(KX)
    return KX


def dHSIC_calc_fast(BatchK:torch.Tensor):
    """
    Calculate the HSIC estimator in the general case d > 2, as in
    [2] Definition 2.6
    """
    #BatchK: DXnXn
    n_k = BatchK.shape[0]

    length = BatchK.shape[1]
    term1 = (1.0/length)**2*(BatchK.cumprod(dim=0)[-1])
    term2 = (1.0/length/length*BatchK).sum(dim=[-1,-2]).cumprod(dim=0)[-1]
    term3 = 2.0/length*((1.0/length*BatchK).sum(dim=1).cumprod(dim=0)[-1])

    term1 = torch.sum(term1)
    term3 = torch.sum(term3)
    dHSIC = term1+term2-term3
    return dHSIC

def dHSIC_fast(argv,sigma=None):
    K_list = batch_gaussian_grammat(argv,sigma=sigma)
    dHSIC =  dHSIC_calc_fast(K_list)
    return dHSIC

if __name__ == '__main__':
    X = 10*torch.randn(85)
    variables = [X + 0.1*torch.randn(85) for i in range(512)]
    variables = torch.stack(variables).cuda()

    hsic = dHSIC_fast(variables,sigma=None)
    print(hsic)
    hsic = dHSIC_fast(variables, sigma=None)
    print(hsic)