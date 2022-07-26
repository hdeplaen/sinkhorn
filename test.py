import torch
from sinkhorn import stable
from ot import sinkhorn
from timeit import timeit

iterMax = 50
N, num_s, num_t = 1, 5, 3
C = torch.abs(torch.randn(N, num_s, num_t))
h_s = torch.ones((N, num_s)) / num_s
h_t = torch.ones((N, num_t)) / num_t

t = timeit()
P = stable(C, h_s, h_t, .1, iterMax)
print(timeit() - t)

t = timeit()
P2 = sinkhorn(h_s.squeeze(), h_t.squeeze(), C.squeeze(), reg=.1, numItermax=iterMax)
print(timeit() - t)

print(P)
print(P2)
