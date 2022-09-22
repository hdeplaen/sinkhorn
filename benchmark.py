import torch
from sinkhorn import stable, base_single, base
from ot import sinkhorn
from timeit import timeit
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

dev = torch.device('cpu')

num_expe = 150
num_expe_timeit = 10
N_exp = np.arange(3,9)
iterMax_max_exp = [7]
results = torch.zeros(len(N_exp),len(iterMax_max_exp),2,num_expe)

stmt0 = 'base_single(h_s, h_t, C, .1, iterMax)'
stmt1 = "sinkhorn(h_s.squeeze(), h_t.squeeze(), C.squeeze(), reg=.1, numItermax=(iterMax+1), log=True,method='sinkhorn')"

for ii in tqdm(range(len(N_exp))):
    for jj in range(len(iterMax_max_exp)):
        for kk in range(num_expe):
            N = 2 ** N_exp[ii]
            iterMax = 2 ** iterMax_max_exp[jj]
            num_s, num_t = 50, 30
            C = torch.abs(torch.randn(num_s, num_t, device=dev))
            h_s = torch.ones((num_s,), device=dev) / num_s
            h_t = torch.ones((N, num_t), device=dev) / num_t

            base_single(h_s, h_t, C, .1, iterMax)
            results[ii,jj,0,kk] = timeit(stmt=stmt0,
                                         setup='pass',
                                         number=num_expe_timeit,
                                         globals={'h_s': h_s,
                                                  'h_t': h_t,
                                                  'C': C,
                                                  'base_single': base_single,
                                                  'iterMax': iterMax})

            h_t = h_t.T
            results[ii,jj,1,kk] = timeit(stmt=stmt1,
                                         setup='pass',
                                         number=num_expe_timeit,
                                         globals= {'h_s': h_s,
                                                  'h_t': h_t,
                                                   'C': C,
                                                  'sinkhorn': sinkhorn,
                                                  'iterMax': iterMax})


res_mean = torch.mean(results,3)
res_std = torch.std(results,3)

## PLOTS
fig, ax = plt.subplots(1)
ax.plot(N_exp, (res_mean[:,0,0]).squeeze(), label='Our implementation', color='red')
ax.plot(N_exp, (res_mean[:,0,1]).squeeze(), label='Python Optimal Transport', color='blue')
ax.fill_between(N_exp, res_mean[:,0,0] + res_std[:,0,0], res_mean[:,0,0] - res_std[:,0,0], facecolor='red', alpha=.3)
ax.fill_between(N_exp, res_mean[:,0,1] + res_std[:,0,1], res_mean[:,0,1] - res_std[:,0,1], facecolor='blue', alpha=.3)
ax.legend()
ax.set_title('Single cost matrix, 1 histogram vs many [CPU]')
ax.set_xlabel('Histogram size')
ax.set_ylabel('Time [s]')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
