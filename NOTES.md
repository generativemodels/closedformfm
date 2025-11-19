

### How to run

####

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

eval "$(python src/train_toy2d.py -sc install=bash)"

uv run src/train_toy2d.py

# or to schedule with slurm
uv run src/train_toy2d.py -m +slurm=cpu

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



### Clean up notes

- [x] use uv
- [x] clarify what is best practice for wconf and hydra, decide folder structure
- [x] import one learning, typically 2D toy data
- [x] example slurm conf etc
- [ ] import a figure code, maybe the histogram one, for 2d toy data
- [ ] import the rest
- [ ] add license
- [ ] add tests

