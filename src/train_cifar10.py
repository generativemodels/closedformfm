# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.
# and https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/train_cifar10.py

import time
import copy
import math
import hydra
import pickle
from omegaconf import OmegaConf


total_num_gpus = 1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

num_workers = 2

compute_fid5k_every = 50_000
compute_fid50k_every = 100_000
batch_size_fid = 124


@hydra.main(
    version_base=None, config_path="../conf", config_name="train_cifar10")
def train(cfg):
    def warmup_lr(step):
        return min(step, warmup) / warmup

    import numpy as np
    import torch
    from torchvision import datasets, transforms
    from tqdm import trange
    from utils_cifar10 import ema, infiniteloop, model_mul
    # generate_samples


    from torchcfm.conditional_flow_matching import (
        ConditionalFlowMatcher,
        ExactOptimalTransportConditionalFlowMatcher,
        TargetConditionalFlowMatcher,
        VariancePreservingConditionalFlowMatcher,
        pad_t_like_x
    )
    from torchcfm.models.unet.unet import UNetModelWrapper
    from utils_metrics import flatten, getall
    from utils_mean_cfm import get_full_velocity_field

    from torchvision.datasets.cifar import CIFAR10
    from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
    from compute_fid import compute_fid

    import mlflow
    from mlflow.models import infer_signature
    print("Training about to start")
    mlflow.set_experiment("cifar10")
    mlflow.start_run()
    run = mlflow.active_run()
    run_id = run.info.run_id
    print("Training started")

    num_channels = getall(cfg.net, 'num_channels')[0]
    batch_size, lr, warmup, ema_decay, ema_start, grad_clip = getall(
        cfg.optimizer, 'batch_size, lr, warmup, ema_decay, ema_start, grad_clip')
    total_steps, dump_models, dump_every, dump_points, nodump_before = getall(
        cfg.trainer, 'total_steps, dump_models, dump_every, dump_points, nodump_before')
    expected_ucond, n_samples_mean, batch_size_mean, sigmamin, model_name, rescaled, tmin, tmax = getall(
        cfg.loss,
        'expected_ucond, n_samples_mean, batch_size_mean, sigmamin, model_name, rescaled, tmin, tmax')
    integration_method, integration_steps = getall(
        cfg.sampler, "integration_method, integration_steps")
    n_subsample, random_horizontal_flip = getall(cfg.data, "n_subsample, random_horizontal_flip")
    print(n_subsample)
    # import ipdb; ipdb.set_trace()
    mlflow.set_tag(
        "mlflow.runName",
        f"{model_name}, E ucond {expected_ucond}, ema {ema_decay}, ema_start {ema_start}, rescaled {rescaled}, bs {batch_size}, lr {lr},  #n_mean {n_samples_mean}")
    print(
        f"lr {lr}, total_steps {total_steps}, ema decay {ema_decay}, dump_every {dump_every}, expected ucond {expected_ucond}, #samples_mean {n_samples_mean}")
    # import ipdb; ipdb.set_trace()
    OmegaConf.set_struct(cfg, False)
    cfg["run_id"] = run_id
    OmegaConf.set_struct(cfg, True)
    mlflow.log_params(flatten(cfg, separator='__'))
    # TODO do better to avoid potential overwriting with multiple process
    with open(f'{run_id}_config_dict.pkl', 'wb') as f:
        pickle.dump(cfg, f)
    mlflow.log_artifact(f'{run_id}_config_dict.pkl', artifact_path="config")

    torch.manual_seed(0)  # TODO better seeding

    if random_horizontal_flip:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    if n_subsample < 50_000:
        dataset.data = dataset.data[:n_subsample]
    batch_size = min(batch_size, n_subsample)

    ft_extractor = InceptionFeatureExtractor(save_path="features")
    train_feat = ft_extractor.get_features(
        CIFAR10(train=True, root="data", download=True), name="cifar10_train")

    sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle= True,
        num_workers=num_workers,
        drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    if expected_ucond:
        dataloader_prime = torch.utils.data.DataLoader(
            dataset,
            batch_size=n_samples_mean,
            sampler=sampler,
            shuffle= True,
            num_workers=num_workers,
            drop_last=True,
        )

        datalooper_prime = infiniteloop(dataloader_prime)

    # Calculate number of epochs
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    num_epochs = math.ceil((total_steps + 1) / steps_per_epoch)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )  # new dropout + bs of 128
    # ).to(
    #         rank
    #     )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    # if parallel:
    #     net_model = DistributedDataParallel(net_model, device_ids=[rank])
    #     ema_model = DistributedDataParallel(ema_model, device_ids=[rank])

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    # sigma = 0.0
    if model_name == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigmamin)
    elif model_name == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigmamin)
    elif model_name == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigmamin)
    elif model_name == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigmamin)
    else:
        raise NotImplementedError(
            f"Unknown {model_name}, must be in ['otcfm', 'icfm', 'fm', 'si']")

    global_step = 0  # to keep track of the global step in training loop

    with trange(num_epochs, dynamic_ncols=True) as epoch_pbar:
        for idx_epoch, epoch in enumerate(epoch_pbar):
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            if sampler is not None:
                sampler.set_epoch(epoch)

            with trange(steps_per_epoch, dynamic_ncols=True) as step_pbar:
                for step in step_pbar:
                    start = time.time()

                    optim.zero_grad()
                    x1 = next(datalooper).to(device)
                    x0 = torch.randn_like(x1)

                    t = torch.rand(x0.shape[0]).type_as(x0)[:, None]
                    t = (tmax - tmin) * t + tmin
                    t_ = pad_t_like_x(t, x0)
                    xt = (1 - t_) * x0 + t_ * x1
                    if expected_ucond and model_name == "icfm":
                        # TODO larger batchsize for x1_prime
                        x1prime = next(datalooper_prime).to(device)
                        x1prime = torch.cat([x1, x1prime])
                        ut = get_full_velocity_field(
                            t, xt, x1prime, sigmamin=sigmamin,
                            batch_size_mean=batch_size_mean, flatten=True)
                    elif (not expected_ucond) and model_name == "icfm":
                        ut = x1 - x0
                    else:
                        t, xt, ut = FM.sample_location_and_conditional_flow(
                            x0, x1)

                    vt = net_model(t, xt)
                    if rescaled:
                        t_ = pad_t_like_x(t, x0)
                        loss = torch.mean(((vt - ut) * (1 - t_)) ** 2)
                    loss = torch.mean((vt - ut) ** 2)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        net_model.parameters(), grad_clip)  # new
                    optim.step()
                    sched.step()

                    # We have done one more step, increment here so that e.g. after 50000 backward, we have global_step%50000 == 0, etc
                    global_step += 1

                    # Update EMA model, ema_start>0 to use corrected one (and skip the first steps)
                    if ema_start == 0: # default ema, actually somewhat wrong, too much weight on the first model
                        ema(net_model, ema_model, ema_decay)
                    else:
                        if global_step == ema_start:
                            unbalanced_ema_model = copy.deepcopy(net_model)
                            model_mul(1-ema_decay, net_model, unbalanced_ema_model) # unb = (1-ema_decay) * M0
                        elif global_step > ema_start:
                            ema(net_model, unbalanced_ema_model, ema_decay)
                            _factor = 1 / (1 - ema_decay**(1 + global_step - ema_start))
                            model_mul(_factor, unbalanced_ema_model, ema_model)
                    end = time.time()

                    with torch.no_grad():
                        if global_step % 100 == 0:
                            mlflow.log_metric(
                                "Training Loss", loss, step=global_step)
                            mlflow.log_metric(
                                "Gradient Norm", grad_norm, step=global_step)
                        print(
                            f"{global_step}: loss {loss.item():0.4f} time {(end - start):0.2f}")
                        if (global_step % compute_fid5k_every == 10):# and global_step > 0:
                        #     num_gen = :
                            # Compute FID 5k online
                            print("FID 5K")
                            num_gen = 5_000
                            fid = compute_fid(ema_model, num_gen, train_feat,
                                              device, ft_extractor, batch_size_fid, integration_method=integration_method, integration_steps=integration_steps)
                            metric_title = f"FID - {(num_gen // 1000)}k {integration_method} {integration_steps} steps"
                            mlflow.log_metric(
                                metric_title, fid, step=global_step)
                        # if (global_step % compute_fid50k_every == 0) and global_step > 0:
                        #     num_gen = 50_000
                        #     fid = compute_fid(
                        #         ema_model, device, num_gen=num_gen,
                        #         batch_size_fid=batch_size_fid,
                        #         integration_method="euler")
                        #     mlflow.log_metric(
                        #         f"FID - {(num_gen // 1000)}k", fid,
                        #         step=global_step)

                        if (np.log2(global_step).is_integer() or global_step % 50_000 == 0) and global_step >= nodump_before:
                            # import ipdb; ipdb.set_trace()
                            signature = infer_signature(
                                x0.cpu().numpy(),
                                ema_model(t, x0).detach().cpu().numpy())
                            mlflow.pytorch.log_model(
                                ema_model, artifact_path=f'model_ema_{global_step}',
                                signature=signature)
                            mlflow.pytorch.log_model(
                                net_model, artifact_path=f'model_{global_step}',
                                signature=signature)

                    # sample and Saving the weights
                    # if dump_every > 0 and global_step % dump_every == 0:
                    #     generate_samples(
                    #         net_model, parallel, savedir, global_step,
                    #         net_="normal"
                    #     )
                    #     generate_samples(
                    #         ema_model, parallel, savedir, global_step,
                    #         net_="ema"
                    #     )
                    #     torch.save(
                    #         {
                    #             "net_model": net_model.state_dict(),
                    #             "ema_model": ema_model.state_dict(),
                    #             "sched": sched.state_dict(),
                    #             "optim": optim.state_dict(),
                    #             "step": global_step,
                    #         },
                    #         savedir + f"{model_name}_cifar10_weights_step_{global_step}.pt",
                    #     )


if __name__ == "__main__":
    train()
