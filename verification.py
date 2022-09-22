import torch
from sinkhorn import stable, base
# from stable import stable
from ot import sinkhorn
from timeit import timeit
from matplotlib import pyplot as plt

iterMax = 200
N, num_s, num_t = 1, 5, 3
C = torch.abs(torch.randn(N, num_s, num_t))
h_s = torch.ones((N, num_s)) / num_s
h_t = torch.ones((N, num_t)) / num_t

t = timeit()
P1 = base(h_s, h_t, C, .1, iterMax)
print(timeit() - t)

t = timeit()
P2, log = sinkhorn(h_s.squeeze(), h_t.squeeze(), C.squeeze(),
                   reg=.1,
                   numItermax=(iterMax+1),
                   log=True,
                   method='sinkhorn_stabilized')
print(timeit() - t)

print(P1)
print(P2)
# print(log['niter'])

fig, axs = plt.subplots(1,2)
axs[0].matshow(P1.squeeze())
axs[1].matshow(P2)
plt.show()

pass
