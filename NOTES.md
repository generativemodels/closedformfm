

### How to run

####

```bash

# trigger installation
uv venv
uv run src/train_toy2d.py


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



# Remove the light=train_light_toy2d to run the full config
uv pip install -e .
python src/train_toy2d.py -m +slurm=cpu
python src/train_toy2d.py -m +slurm=jzv100
```
####

```bash
ssh -L 3839:localhost:31333 labslurm

tmux a
srun -p LONG -t 6-00:00:00  -c 2 --mem=10G  --pty bash -i

ssh -NT -i ~/.ssh/id_ed25519_betweencluster -oExitOnForwardFailure=yes -R 31333:localhost:3999 calcul-slurm-lahc-2 &
trap "kill $!" EXIT
cd prigml
. .venv/bin/activate
fg # for the password...
# Ctrl+Z, bg
mlflow server --port 3999
```



### WIP README.md

Testing the project with minimal "light" configs:

```bash
uv run src/train_toy2d.py +light=train_light_toy2d
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

