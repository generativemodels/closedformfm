

### How to run

####

```bash

# trigger installation
uv venv
uv run src/train_toy2d.py --help


# activate the virtual environment, to used directly python without uv
source .venv/bin/activate
# for autocompletion with python ...
eval "$(python src/train_toy2d.py -sc install=bash)"



python src/train_toy2d.py +light=train_light_toy2d

# or to schedule with slurm (from the frontend node)
# - need "-m" for multirun, which is needed for the slurm launcher, even for a single run
# - IMPORTANT: need to pip install so the slurm launcher finds the package correctly
uv pip install -e .
python src/train_toy2d.py -m +light=train_light_toy2d +slurm=cpu
# or for JZ
python src/train_toy2d.py -m +light=train_light_toy2d +slurm=jzv100

## ALL
uv pip install -e .
python src/train_toy2d.py   -m +light=train_light_toy2d   +slurm=gpu24
python src/train_cifar10.py -m +light=train_light_cifar10 +slurm=gpu24



# Remove the light=train_light_toy2d to run the full config
uv pip install -e .
python src/train_toy2d.py -m +slurm=cpu
python src/train_toy2d.py -m +slurm=jzv100
...

```



#### MLflow server (with back tunneling)

Connect and tunnel:

```bash
# IF YOU HAVE A CONTROL CONNECTION, start the tunnel with:
ssh -O forward -L 3839:localhost:31333 joecluster
ssh joecluster

# IF NOT, CONNECT AND TUNNEL
ssh -L 3839:localhost:31333 joecluster
```

Schedule, back tunnel and start the mlflow server on the cluster (to avoid loading the frontend node):

```bash
tmux a
srun -p LONG -t 6-00:00:00  -c 2 --mem=10G  --pty bash -i
# THEN
ssh -NT -i ~/.ssh/id_ed25519_betweencluster -oExitOnForwardFailure=yes -R 31333:localhost:3999 calcul-slurm-lahc-2 &
trap "kill $!" EXIT
cd closedformfm
. .venv/bin/activate
fg # for the password...
# Ctrl+Z, bg
mlflow server --backend-store-uri sqlite:///$HOME/mlflow.db --default-artifact-root $HOME/mlflow-artifacts --port 3999
```



#### OLD RANDOM

```bash
ssh -L 3839:localhost:31333 labslurm

tmux a
srun -p LONG -t 6-00:00:00  -c 2 --mem=10G  --pty bash -i

ssh -NT -i ~/.ssh/id_ed25519_betweencluster -oExitOnForwardFailure=yes -R 31333:localhost:3999 calcul-slurm-lahc-2 &
trap "kill $!" EXIT
cd closedformfm
. .venv/bin/activate
fg # for the password...
# Ctrl+Z, bg
mlflow server --backend-store-uri sqlite:///$HOME/mlflow.db --default-artifact-root $HOME/mlruns --port 3999


http://locatlhost:3839
```




### Clean up notes

- [x] use uv
- [x] clarify what is best practice for wconf and hydra, decide folder structure
- [x] import one learning, typically 2D toy data
- [x] example slurm conf etc
- [ ] import a figure code, maybe the histogram one, for 2d toy data
- [ ] import the rest
- [ ] add license
- [ ] add tests

