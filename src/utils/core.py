
# Tools with minimal or on-demand imports
import functools
from collections.abc import MutableMapping

# decorator
def hydra_main(config_name, no_runname=False, config_path="../conf", version_base=None):
    def decorator(func):
        import hydra
        @hydra.main(
            config_name=config_name,
            config_path=config_path,
            version_base=version_base
        )
        @functools.wraps(func)
        def wrapper(cfg, *args, **kwargs):
            if not no_runname:
                print(f"## run_name: {cfg.run_name}")
            return func(cfg, *args, **kwargs)
        return wrapper
    return decorator


def mlflow_start(cfg, project_name: str, no_runname=False, no_runid=False, no_params=False, no_force_location=False):
    import mlflow
    from omegaconf import OmegaConf
    from omegaconf import open_dict
    import os
    k = "FORCE_MLFLOW_ARTIFACTS_DESTINATION"
    artifact_location = os.getenv(k, None)
    print(f"## MLFlow experiment: {project_name}")
    print(f"## MLFlow artifact location env var {k}: {artifact_location}")
    if not no_force_location and artifact_location is not None:
        existing_experiment = mlflow.get_experiment_by_name(project_name)
        print(f"## Existing experiment: {existing_experiment}")
        print(f"## Existing artifact location: {existing_experiment.artifact_location if existing_experiment is not None else 'N/A'}")
        need_change = existing_experiment is None or existing_experiment.artifact_location != artifact_location
        if existing_experiment is not None and need_change:
            print(f"## MLFlow disallows changing artifact location of already existing experiment. Experiment ({project_name} already exists, not creating a new one).")
            exit()
        if need_change:
            print(f"## Using forced MLflow artifact location from env var {k}: {artifact_location}")
            mlflow.create_experiment(project_name, artifact_location=artifact_location)

    mlflow.set_experiment(project_name)
    mlflow.start_run()
    run = mlflow.active_run()
    run_id = run.info.run_id
    if not no_runname:
        mlflow.set_tag("mlflow.runName", cfg.run_name)
    if not no_runid:
        with open_dict(cfg):
            cfg.run_id = run_id
    if not no_params:
        mlflow.log_params(flatten(cfg, separator='__'))
    return run



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

