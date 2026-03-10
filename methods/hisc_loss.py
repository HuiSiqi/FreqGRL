import torch

def hisc(Kx,Ky):
    Kxy = Kx@Ky
    n = Kxy.shape[0]
    h = Kxy.trace()