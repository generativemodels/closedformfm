import torch
import matplotlib.pyplot as plt


def sample_gaussian(batch_size, sigma=1):
    return sigma * torch.randn(batch_size, 2)


def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon


def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0


def plot_trajectories(traj, train=None):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(
        traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black",
        label="Prior sample z(S)")
    plt.scatter(
        traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive",
        label="Flow")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    if train is not None:
        plt.scatter(train[:, 0], train[:, 1], c="red", label="Training Set",
                    s=4, alpha=.2)
    plt.legend()
    # plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([-5, 5])
    plt.yticks([-5, 5])
    plt.show(block=False)
