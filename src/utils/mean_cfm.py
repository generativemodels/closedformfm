import tqdm
import torch
import torch.nn.functional as F


def solve_ode_t(xt, tstart=0., n_steps=100, use_u_theta=False, model=None,
                x1prime=None):
    device = xt.device
    traj = torch.zeros((n_steps+1,) + xt.shape).to(device)
    traj[0] = xt.clone()
    # for idx in tqdm.trange(n_steps):
    for idx in tqdm.trange(n_steps):
        t = (tstart + (idx/n_steps) * (1-tstart))
        t_ = t * torch.ones(xt.shape[0], 1).type_as(xt).to(device)
        if use_u_theta:
            with torch.no_grad():
                ut = model(t_, traj[idx])
        else:
            ut = get_full_velocity_field(
                t_,
                traj[idx], x1prime, batch=True, batch_size_mean=16, flatten=True)
        traj[idx+1] = traj[idx] + (1.-tstart)/n_steps * ut
    return traj


def get_log_pt(t, xt_flatten, x1prime_flatten):
    img_dim = xt_flatten.shape[-1]
    # n_samples_mean * batch_size * img_dim
    mut = t[None, :, :] * x1prime_flatten[:, None, :]
    # 1 * batch_size
    sigmat = 1. - t[None, :, 0]
    # n_samples_mean * batch_size
    logprob = -((xt_flatten - mut) ** 2).sum(dim=-1) / (2 * sigmat ** 2)
    # batchsize
    logprob = torch.logsumexp(logprob, dim=0)
    logprob -= 0.5 * torch.log(2 * torch.tensor(torch.pi))
    logprob -= torch.log(sigmat[0]) * img_dim
    # MM TODO the normalization constant here seems wrong, d should be multiplying
    # the 0.5 log(2 pi) term
    logprob -= torch.log(torch.tensor(x1prime_flatten.shape[0]))
    return logprob


def closest_image(x, dataset, index=False):
    "Return closest image to x in dataset."
    # x: bs * channels * height * width
    cdist = torch.cdist(
        torch.flatten(x, start_dim=1),
        torch.flatten(dataset, start_dim=1)
    )
    min_data = torch.min(cdist, dim=1)
    argmin = min_data.indices
    img = dataset[argmin]
    dist = min_data.values
    if index:
        return img, dist, argmin
    else:
        return img


def get_full_velocity_field(
        t, xt, x1prime, sigmamin=.0, batch_size_mean=64, batch=True,
        optimized=False, flatten=True):
    if flatten:
        sizes = xt[0].shape
        xt = torch.flatten(xt, start_dim=1)
        x1prime = torch.flatten(x1prime, start_dim=1)
    if optimized:
        ut = get_full_velocity_field_optimized_(
            t, xt, x1prime, sigmamin=sigmamin)
    elif batch:
        ut = get_full_velocity_field_batch(
            t, xt, x1prime, sigmamin=sigmamin, batch_size_mean=batch_size_mean)
    else:
        ut = get_full_velocity_field_(t, xt, x1prime, sigmamin=sigmamin)
    if flatten:
        ut = torch.unflatten(ut, dim=1, sizes=sizes)
    return ut


def get_full_velocity_field_(t, xt, x1prime, sigmamin=.0):
    """
    Parameters:
    t: batch_size
    x: batch_size
    x1: batch_size

    Returns:
        torch array: (batch_size * size_img)
    -------
    """
    # n_samples_mean * batch_size * size_img
    mut = t[None, :, :] * x1prime[:, None, :]
    sigmat = (1 - (1 - sigmamin) * t)[None, :, :]
    # n_samples_mean * batch_size * size_img
    arg_softmax = mut - xt[None, :, :]
    # TODO batch this sum
    arg_softmax = - ((arg_softmax) ** 2).sum(dim=-1, keepdims=True)
    # n_samples_mean * batch_size
    arg_softmax /= 2 * sigmat ** 2

    pcond = F.softmax(arg_softmax, dim=0)
    ucond = (x1prime[:, None, :] - (1 - sigmamin) * xt[None, :, :]) / sigmat
    utot = (ucond * pcond).sum(dim=0)
    import ipdb; ipdb.set_trace()
    return utot


def get_full_velocity_field_optimized_(t, xt, x1prime, sigmamin=.0):
    """
    Parameters:
    t: batch_size * 1
    xt: batch_size * size_img
    x1prime: batch_size_mean * size_img

    Returns:
        torch array: (batch_size * size_img)
    -------
    """
    # n_samples_mean * batch_size * size_img
    t = t.reshape(-1,)
    sigmat = (1 - (1 - sigmamin) * t)
    arg_softmax = (xt ** 2).sum(dim=-1)[None, :]
    # batch_size_mean * batch_size
    arg_softmax = arg_softmax + (t ** 2) * (x1prime ** 2).sum(dim=-1)[:, None]
    arg_softmax += - 2 * t[None, :] * (x1prime @ xt.T)
    arg_softmax = - arg_softmax
    arg_softmax /= (2 * sigmat ** 2)[None, :]

    pcond = F.softmax(arg_softmax, dim=0)
    utot = ((pcond.T @ x1prime) - (1 - sigmamin) * xt)
    utot = utot / sigmat[:, None]
    return utot


def get_full_velocity_field_batch(
        t, xt, x1prime, sigmamin=.0, batch_size_mean=64):
    """
    Parameters:
    t: batch_size
    x: batch_size
    x1: batch_size

    Returns:
        torch array: (batch_size * img_size)
    -------
    """
    n_samples_mean = x1prime.shape[0]
    batch_size = xt.shape[0]
    arg_softmax = torch.zeros(
        (n_samples_mean, batch_size, 1), device=t.device)
    # import ipdb; ipdb.set_trace()
    # n_samples_mean * batch_size * img_size
    # mut =
    # n_samples_mean * batch_size * img_size
    # Here I batched along n_samples_mean
    # Should we batch along dimension?
    if n_samples_mean % batch_size_mean == 0:
        n_iter = n_samples_mean // batch_size_mean
    else:
        n_iter = n_samples_mean // batch_size_mean + 1

    for batch in range(n_iter):
        idx0 = batch * batch_size_mean
        idx1 = (batch + 1) * batch_size_mean
        if batch == (n_samples_mean // batch_size_mean):
            idx1 = n_samples_mean
        mut = t[None, :, :] * x1prime[idx0:idx1, None, :]
        arg_softmax_ = mut - xt[None, :, :]
        arg_softmax[idx0:idx1, :, :] = - (
            (arg_softmax_) ** 2).sum(dim=-1, keepdims=True)
        # n_samples_mean * batch_size
    sigmat = (1 - (1 - sigmamin) * t)[None, :, :]
    arg_softmax /= 2 * sigmat ** 2
    pcond = F.softmax(arg_softmax, dim=0)

    utot = torch.zeros_like(xt)

    # import ipdb; ipdb.set_trace()
    # Batching along n_samples_mean requires 2 loops because of the softmax
    for batch in range(n_iter):
        idx0 = batch * batch_size_mean
        idx1 = (batch + 1) * batch_size_mean
        if batch == (n_samples_mean // batch_size_mean):
            idx1 = n_samples_mean
        ucond = (x1prime[idx0:idx1, None, :] - (1 - sigmamin) * xt[None, :, :]) / sigmat
        utot += (ucond * pcond[idx0:idx1, :, :]).sum(dim=0)
    return utot
