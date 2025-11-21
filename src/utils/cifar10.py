"""Taken from
https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/utils_cifar.py
"""

import copy
import os

import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

def model_mul(scale, source, target): # scale*source -> target
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(source_dict[key].data * scale)


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

import ot
# import numpy as np

def w_pairing(x0, x1):
    assert x0.shape[0] != x0.shape[1]
    n_gen_samples = x0.shape[0]
    ab = torch.ones(n_gen_samples) / n_gen_samples
    # train_samples = train_set[:n_gen_samples, :].numpy()
    M = ot.dist(x0, x1)
    ot_matrix = ot.emd(ab, ab, M)
    # pairing_matrix = np.where(log["G"])
    return torch.argmax(ot_matrix, axis=0)


class PairingDataset(Dataset):
    """Pairing dataset."""

    def __init__(self, base_dataset, compute_pairing):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_len = len(base_dataset)
        self.base_dataset = base_dataset

        self.noise = torch.randn([self.dataset_len, *base_dataset[0][0].shape])
        x0 = torch.flatten(self.noise, start_dim=1)
        x1 = torch.flatten(
            torch.stack([xy[0] for xy in self.base_dataset]), start_dim=1)
        # if False:
        #     # check direction
        #     mapping = compute_pairing(x0, x1)
        #     print(((x1 - x0[mapping])**2).sum())
        #     mapping = compute_pairing(x1, x0)
        #     print(((x1 - x0[mapping])**2).sum())
        x0_to_x1 = compute_pairing(x0, x1)
        self.noise = self.noise[x0_to_x1]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.base_dataset[idx], self.noise[idx]


if __name__ == "__main__":
    from torchvision import datasets, transforms
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    # import ipdb; ipdb.set_trace()
    # from utils_dinov2 import DINOv2FeatureExtractor
    # feature_extractor = DINOv2FeatureExtractor("cpu")

    # small_dataset = torch.stack([dataset[index][0] for index in range(5000)])
    # pairing = PairingDataset(small_dataset, w_pairing)


    import torch
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    # from dinov2.models import dinov2_vits14

    # Set up the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load the pre-trained DINOv2 model
    # model = dinov2_vits14()
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()

    # Function to extract embeddings
    def extract_embeddings(model, data_loader):
        embeddings = []
        labels = []
        with torch.no_grad():
            for images, label in data_loader:
                features = model(images)
                embeddings.append(features)
                labels.append(label)
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)
        return embeddings, labels

    # Extract embeddings for the CIFAR-10 dataset
    embeddings, labels = extract_embeddings(model, train_loader)

    # Now you can use the embeddings for further tasks like clustering or classification
    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)
