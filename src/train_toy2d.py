
from utils.core import hydra_main, mlflow_start

# delay most imports to have faster access to hydra config from the command line

@hydra_main("train_toy2d")
def train(cfg):

    import time
    import tempfile
    from pathlib import Path
    import numpy as np
    import torch
    import mlflow
    from mlflow.models import infer_signature

    from torchcfm.utils import sample_8gaussians
    from torchcfm.optimal_transport import OTPlanSampler

    from models.simple import MLP
    from utils.otcfm import (
        compute_conditional_vector_field, sample_conditional_pt, sample_gaussian)
    from utils.mean_cfm import get_full_velocity_field
    from utils.metrics import get_gen_samples, get_w_dist

    # Create a new MLflow Experiment (and configure it automatically)
    mlflow_start(cfg, "toy2d")

    dim = cfg.net.dim
    n_layers = cfg.net.n_layers
    w = cfg.net.w
    decoupled = cfg.net.decoupled
    batch_size = cfg.optimizer.batch_size
    lr = cfg.optimizer.lr
    n_epochs = cfg.trainer.n_epochs
    dumps_models = cfg.trainer.dump_models
    dump_every = cfg.trainer.dump_every
    dump_points = cfg.trainer.dump_points
    integration_method = cfg.sampler.integration_method
    integration_steps = cfg.sampler.integration_steps
    expected_ucond = cfg.loss.expected_ucond
    batch_size_mean = cfg.loss.batch_size_mean
    sigmamin = cfg.loss.sigmamin
    use_ot_sampler = cfg.loss.use_ot_sampler
    rescaled = cfg.loss.rescaled
    seed = cfg.data.seed
    n_gen_samples = cfg.data.n_gen_samples


    torch.manual_seed(seed)  # TODO better seeding (across np, torch etc)

    x1_train = sample_8gaussians(n_gen_samples)
    x1_test = sample_8gaussians(n_gen_samples)
    w_dist_train_test = get_w_dist(x1_train.numpy(), x1_test.numpy())
    print(f"Wasserstein(Train, Test):{w_dist_train_test}")
    # keep x0 and x1 fixed for test to avoid noise
    x0_test = sample_gaussian(n_gen_samples)

    # Dirty check
    # np.testing.assert_allclose(x1_train.sum().numpy().sum(), 53.56688)

    model = MLP(
        w=w, dim=dim, time_varying=True, n_layers=n_layers,
        decoupled=decoupled, sigmamin=sigmamin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ot_sampler = OTPlanSampler(method="exact")

    start = time.time()
    for current_epoch in range(n_epochs+1):
        optimizer.zero_grad()

        x0 = sample_gaussian(batch_size)

        idx = torch.randint(0, n_gen_samples, (batch_size,))
        x1 = x1_train[idx, :]

        if use_ot_sampler:
            x0, x1 = ot_sampler.sample_plan(x0, x1)

        t = torch.rand(x0.shape[0]).type_as(x0)[:, None]

        xt = sample_conditional_pt(x0, x1, t, sigma=sigmamin)
        if expected_ucond:
            idxprime = torch.randint(0, n_gen_samples, (batch_size_mean,))
            x1prime = torch.cat([x1, x1_train[idxprime]])
            ut = get_full_velocity_field(t, xt, x1prime, sigmamin=sigmamin)
        else:
            ut = compute_conditional_vector_field(x0, x1)
        vt = model(torch.cat([xt, t], dim=-1))
        # TODO better name than rescaled
        if rescaled:
            loss = torch.mean(((vt - ut) * (1 - (1 - sigmamin) * t)) ** 2)
        else:
            loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        # TODO add option for gradient clipping
        grad_clip = 1_000_000
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip)
        optimizer.step()
        end = time.time()

        mlflow.log_metric("Training Loss", loss, step=current_epoch)
        mlflow.log_metric("Gradient Norm", grad_norm, step=current_epoch)
        if current_epoch % 100 == 0:
            print(
                f"{current_epoch}: loss {loss.item():0.4f} time {(end - start):0.2f}")

        if current_epoch % 10 == 0:
            with torch.no_grad():
                print(
                    f"{current_epoch}: loss {loss.item():0.3f} time {(end - start):0.2f}")
                ut = get_full_velocity_field(
                    t, xt, x1_train, sigmamin=sigmamin)
                velocity_matching_loss = torch.mean((vt - ut) ** 2)
                mlflow.log_metric(
                    "VM Loss", velocity_matching_loss, step=current_epoch)

        if (current_epoch % dump_every == 0) and current_epoch != 0:
            with torch.no_grad():
                gen_samples = get_gen_samples(
                    model, x0_test, solver=integration_method, n_steps=integration_steps)
                # Compute Wasserstein distance
                w_dist_train = get_w_dist(gen_samples, x1_train.numpy())
                w_dist_test = get_w_dist(gen_samples, x1_test.numpy())
                mlflow.log_metric("W train", w_dist_train, step=current_epoch)
                mlflow.log_metric("W test", w_dist_test, step=current_epoch)
                print(f"W train: {w_dist_train:0.3f}")
                print(f"W test: {w_dist_test:0.3f}")

                filename = f"samples-{current_epoch:06d}.npz"
                if dump_points:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        path = Path(tmp_dir, filename)
                        np.savez(path, gen_samples=gen_samples, x1_train=x1_train, x1_test=x1_test, x0=x0)
                        mlflow.log_artifact(path)

        if (dumps_models & (current_epoch % dump_every == 0)) or current_epoch == n_epochs:
            with torch.no_grad():
                signature = infer_signature(
                    x0.numpy(),
                    model(torch.cat([x0, t], dim=-1)).detach().numpy())
                mlflow.pytorch.log_model(
                    model, name=f'training-state-{current_epoch}',
                    signature=signature)

    mlflow.end_run()



if __name__ == "__main__":
    train()
