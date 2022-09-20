import torch
from sinkhorn import stable, base
from ot import sinkhorn
from timeit import timeit
from matplotlib import pyplot as plt

num_expe = 10
results = torch.zeros(8,15,2,num_expe)

for ii in range(8):
    for jj in range(15):
        for kk in range(num_expe):
            N = 2 ** (ii+1)
            iterMax = 2 ** jj
            num_s, num_t = 50, 30
            C = torch.abs(torch.randn(N, num_s, num_t))
            h_s = torch.ones((N, num_s)) / num_s
            h_t = torch.ones((N, num_t)) / num_t

            t = timeit()
            P1 = stable(h_s, h_t, C, .1, iterMax)
            results[ii,jj,0,kk] = timeit() - t

            C = C.moveaxis(0,2)
            h_s = h_s.moveaxis(0,1)
            h_t = h_t.moveaxis(0,1)

            t = timeit()
            P2, log = sinkhorn(h_s.squeeze(), h_t.squeeze(), C.squeeze(),
                   reg=.1,
                   numItermax=(iterMax+1),
                   log=True,
                   method='sinkhorn')
            results[ii,jj,1,kk] = timeit() - t


print(results)
