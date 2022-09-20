import torch


def _update_K(alpha, beta, C, reg):
    return torch.exp(-(C - alpha.unsqueeze(2).expand_as(C)
                       - beta.unsqueeze(1).expand_as(C)) / reg)


def _get_P(alpha, beta, u, v, C, reg):
    return torch.exp(-((C - alpha.unsqueeze(2).expand_as(C)
                        - beta.unsqueeze(1).expand_as(C)) / reg)
                     + u.log().unsqueeze(2).expand_as(C)
                     + v.log().unsqueeze(1).expand_as(C))


def stable(h_s, h_t, C, reg, numIter):
    N = C.size(0)
    num_s = C.size(1)
    num_t = C.size(2)

    u = torch.ones(N, num_s) / num_s
    v = torch.ones(N, num_t) / num_t
    alpha = torch.zeros(N, num_s)
    beta = torch.zeros(N, num_t)
    K = _update_K(alpha, beta, C, reg)

    for i in range(numIter):
        u = h_s / ((K * v.unsqueeze(1)).sum(2))
        v = h_t / ((K * u.unsqueeze(2)).sum(1))

        alpha = alpha + reg * torch.max(torch.log(u), dim=0, keepdim=True)[0]
        beta = beta + reg * torch.max(torch.log(v), dim=0, keepdim=True)[0]

        u.fill_(1. / num_s)
        v.fill_(1. / num_t)

        K = _update_K(alpha, beta, C, reg)
    u = h_s / ((K * v.unsqueeze(1)).sum(2))
    v = h_t / ((K * u.unsqueeze(2)).sum(1))
    return _get_P(alpha, beta, u, v, C, reg)
