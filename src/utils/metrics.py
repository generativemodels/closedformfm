import torch
import ot
import os
import tqdm
import numpy as np
from torchvision.utils import save_image
from collections.abc import MutableMapping
from torchdyn.core import NeuralODE
from torchdiffeq import odeint
from cleanfid import fid


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, eps=1e-5):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        val = self.model(torch.cat(
            [x, t.repeat(x.shape[0])[:, None]], 1))
        return val


def get_gen_samples(model, x0, solver="dopri5", n_steps=100):
    node = NeuralODE(
        torch_wrapper(model), solver=solver, sensitivity="adjoint",
        atol=1e-4, rtol=1e-4)
    traj = node.trajectory(
        x0,
        t_span=torch.linspace(0, 1, n_steps),
    )
    gen_samples = traj[-1, :, :].cpu().numpy()  # TODO: stay on GPU?
    return gen_samples


def get_w_dist(gen_samples, train_samples):
    n_gen_samples = gen_samples.shape[0]
    ab = np.ones(n_gen_samples) / n_gen_samples
    # train_samples = train_set[:n_gen_samples, :].numpy()
    M = ot.dist(gen_samples, train_samples)
    wasserstein_dist = ot.emd2(ab, ab, M)
    return wasserstein_dist


def flatten(dictionary, parent_key='', separator='_'):
    """ Code Taken from
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def getall(cfg, keys):
    import re
    return [cfg[key] for key in re.split(r'\s*,\s*', keys)]


def generate_samples(
        network, device, path=None, integration_method="dopri5", tol=1e-5,
        n_samples=1028, batch_size=None, num_channels=3, res=32, n_classes=10,
        integration_steps=100, tmax=1):
    """
    Return a tensor of size (TODO).
    """

    if path is not None:
        os.makedirs(path, exist_ok=True)

    if batch_size is None:
        batch_size = n_samples

    images_list = []
    batches = [batch_size] * (n_samples // batch_size)
    if n_samples % batch_size:
        batches += [n_samples % batch_size]

    with torch.no_grad():
        for k, batch in enumerate(tqdm.tqdm(batches)):
            time_points = torch.linspace(
                0, tmax, int(tmax * integration_steps), device=device)

            x0 = torch.randn(batch, num_channels, res, res, device=device)
            traj = odeint(
                network, x0, time_points, rtol=tol, atol=tol,
                method=integration_method)
            if path is None:
                images_list.append(traj[-1, :])
            else:
                rescaled_images = traj[-1, :].view(
                    [-1, num_channels, res, res]).clip(-1, 1)
                rescaled_images = rescaled_images / 2 + 0.5
                # rescaled_images = (traj[-1, :] * 127.5 +
                #                    128).clip(0, 255)  # .to(torch.uint8)
                for i, img in enumerate(rescaled_images):
                    save_image(img, path + f"image_{batch_size *  k  + i}.png")

    if path is None:
        images = torch.cat(images_list, dim=0)
        return images


# def generate_and_save_samples(
#         network, savedir, step, device, integration_method="dopri5",
#         n_samples=100,
#         tol=1e-5, res=32, num_channels=3):
#     """
#     Return path to the generated images, clipped between 0 and 1.
#     """
#     network.eval()
#     path = os.path.join(savedir, f"ema_generated_FM_images_step_{step}.png")
#     images = generate_samples(
#             network, device, integration_method=integration_method, tol=tol,
#             n_samples=n_samples, res=res, num_channels=num_channels,
#             conditional_embedding=conditional_embedding,
#             conditional_sampling=conditional_sampling)
#     rescaled_images = images.view([-1, num_channels, res, res]).clip(-1, 1)
#     rescaled_images = rescaled_images / 2 + 0.5
#     save_image(rescaled_images, path, nrow=10)
#     network.train()
#     return path


# def compute_fid(
#         network, device, integration_method="dopri5", integration_steps=100,
#         tol=1e-5, num_gen=50_000, batch_size_fid=1020):
#     network.eval()

#     def gen_batch_img(unused_latent):
#         images = generate_samples(
#             network, device, integration_method=integration_method,
#             integration_steps=integration_steps, tol=tol,
#             n_samples=batch_size_fid)
#         # resizing and clipping between 0 and 255 for clean fid
#         rescaled_images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
#         # .permute(1, 2, 0)
#         return rescaled_images

#     print("Start computing FID")
#     score = fid.compute_fid(
#         gen=gen_batch_img,
#         dataset_name="cifar10",
#         batch_size=batch_size_fid,
#         dataset_res=32,
#         num_gen=num_gen,
#         dataset_split="train",
#         mode="legacy_tensorflow",
#     )
#     print("FID has been computed")
#     print("FID: %.2f on %ik samples" % (score, num_gen // 1_000))
#     network.train()
#     return score


# if __name__ == "__main__":
#     # Test if compute FID is working
#     from torchcfm.models.unet.unet import UNetModelWrapper
#     # Define the model
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")

#     new_net = UNetModelWrapper(
#         dim=(3, 32, 32),
#         num_res_blocks=2,
#         num_channels=128,
#         channel_mult=[1, 2, 2, 2],
#         num_heads=4,
#         num_head_channels=64,
#         attention_resolutions="16",
#         dropout=0.1,
#     ).to(device)
#     # Load the model
#     PATH = "/network/scratch/q/quentin.bertrand/perfgen/experiments/cifar_cfm_pretrain_pregen/otcfm_cifar10_weights_step_400000.pt"
#     # PATH = f"{FLAGS.input_dir}/{FLAGS.model}/{FLAGS.model}_cifar10_weights_step_{FLAGS.step}.pt"
#     print("path: ", PATH)
#     checkpoint = torch.load(PATH)
#     state_dict = checkpoint["ema_model"]
#     try:
#         new_net.load_state_dict(state_dict)
#     except RuntimeError:
#         from collections import OrderedDict

#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             new_state_dict[k[7:]] = v
#         new_net.load_state_dict(new_state_dict)
#     new_net.eval()
#     compute_fid(
#         new_net, device, integration_method="euler", tol=1e-5, num_gen=5_000)
#     # compute_fid(
#     #     new_net, device, integration_method="dopri5", tol=1e-5, num_gen=5_000)
