import torch
import numpy as np
import os
import hydra
import mlflow
import mlflow.pytorch as mlpt
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.celeba import CelebA
from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FID import FID
from mlflow import MlflowClient
from torchvision import datasets, transforms

from utils.metrics import generate_samples

EXTRACTORS = {"inception": InceptionFeatureExtractor, "dino": DINOv2FeatureExtractor}


def compute_fid(model, num_images_fid, train_feat, device, ft_extractor, batch_size=512, integration_method="dopri5", integration_steps=100, res=32):
    gen_images = generate_samples(model, device, integration_method=integration_method, tol=1e-4,
                                  n_samples=num_images_fid, res=res, batch_size=batch_size, integration_steps=integration_steps)
    gen_images = (gen_images * 127.5 + 128).clip(0, 255).to(torch.uint8)
    gen_feat = ft_extractor.get_tensor_features(
        gen_images)
    fid_val = FID().compute_metric(
        train_feat, None, gen_feat)
    return fid_val


def compute_fid_from_dir(gen_path, num_images_fid, train_feat, device, ft_extractor):
    gen_feat = ft_extractor.get_dir_features(gen_path)
    fid_val = FID().compute_metric(
        train_feat, None, gen_feat)
    return fid_val


@hydra.main(version_base=None, config_path="conf", config_name="config_metric")
def log_fid(cfg):
    # run_id = "8efb498ce85e4afab1d3292a4a3674c7"
    run_id = cfg.run_id
    ema = cfg.generation.ema

    save_path = (f"samples/{run_id}/solver_{cfg.generation.solver}/"
                 f"inte_steps_{cfg.generation.steps}/ema_{cfg.generation.ema}/"
                 f"{cfg.fid.num_images_fid//1000}k")

    mlflow.start_run(run_id)
    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    ft_extractor = EXTRACTORS[cfg.fid.ft_extractor](save_path="features")

    if cfg.dataset.lower() == "cifar10":
        train_feat = ft_extractor.get_features(
            CIFAR10(train=True, root="data", download=True), name="cifar10_train")
        test_feat = ft_extractor.get_features(
            CIFAR10(train=False, root="data", download=True), name="cifar10_test")
        res = 32
    elif cfg.dataset.lower() == "celeba":
        train_feat = ft_extractor.get_features(
            CelebA(split='train', root="../data", download=True,     transform=transforms.Compose(
                [transforms.CenterCrop(178),
                 transforms.Resize([64, 64]),])), name="celeba64_train")
        test_feat = ft_extractor.get_features(
            CelebA(split='test', root="../data", download=True, transform=transforms.Compose(
                [transforms.CenterCrop(178),
                 transforms.Resize([64, 64]),])), name="celeba64_test")
        res = 64
    elif cfg.dataset.lower() == "tiny_imagenet":
        from tinyimagenet import TinyImageNet # import on demand to get optional dependency
        train_feat = ft_extractor.get_features(
            TinyImageNet(split='train', root="./data/tinyimagenet",    transform=transforms.Compose(
                [transforms.CenterCrop(178),
                 transforms.Resize([64, 64]),])), name="tiny_train")
        test_feat = ft_extractor.get_features(
            TinyImageNet(split='test', root="./data/tinyimagenet", transform=transforms.Compose(
                [transforms.CenterCrop(178),
                 transforms.Resize([64, 64]),])), name="tiny_test")
        res = 64

    else:
        raise ValueError()
    # test_feat = ft_extractor.get_features(
    # CIFAR10(train=False, root="data", download=True))

    num_images_fid = cfg.fid.num_images_fid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_steps = cfg.generation.global_steps
    print(f"{global_steps=}")
    if len(global_steps) == 0:
        global_steps = []
        # we expect model and model_ema to be saved at same global steps:
        for artifact in artifacts:
            if not artifact.path.startswith("model_"):
                continue
            else:
                if not artifact.path.startswith("model_ema"):
                    global_steps.append(int(artifact.path[len("model_"):]))
        global_steps = np.sort(global_steps)

    for global_step in global_steps:
        print("k", global_step)
        path = f"runs:/{run_id}/model_{'ema_'*ema}{global_step}"

        gen_path = save_path + f"/step_{global_step}/"
        # fid_val = compute_fid(model, num_images_fid, train_feat, device, ft_extractor)
        # be carfefull to the number of samples in the dir
        if os.path.isdir(gen_path) and len(os.listdir(gen_path)) == num_images_fid:
            pass
        else:
            model = mlpt.load_model(path)
            model.eval()
            model.to(device)
            print(f"Generating {num_images_fid} samples for FID")
            generate_samples(model, device, path=gen_path, integration_method=cfg.generation.solver, tol=1e-4,
                             n_samples=num_images_fid, batch_size=cfg.generation.batch_size_gen, integration_steps=cfg.generation.steps, res=res)

        for data_name, feat in zip(["train", "test"], [train_feat, test_feat]):
            fid = compute_fid_from_dir(
                gen_path, num_images_fid, feat, device, ft_extractor)
            metric_name = (
                f"FID {data_name} {cfg.fid.ft_extractor} {cfg.fid.num_images_fid // 1000}k ema  {ema}")
            mlflow.log_metric(metric_name, fid, step=global_step)
        # del model

    mlflow.end_run()


if __name__ == "__main__":
    log_fid()
