import datetime
import pathlib
import shutil


# TODO: Add default path param
def init_model_dir(base_dir: pathlib.Path) -> pathlib.Path:
    """
    Create a new timestamped model dir.

    Parameters
    ----------
    base_dir : pathlib.Path
        Base model path. The current timestamp will be appended to that base.

    Returns
    -------
    pathlib.Path
        Timestamped model dir path.

    Examples
    --------
    >>> init_model_dir(pathlib.Path('/path/to/model'))
    /path/to/model.20220518-010621
    """
    model_dir = base_dir.with_suffix(datetime.datetime.now().strftime(".%Y%m%d-%H%M%S"))
    model_dir.mkdir(parents=True, exist_ok=False)
    print(f"Model directory: {model_dir}")
    return model_dir


# TODO: Add default path param
def init_cache_dir(cache_dir: pathlib.Path) -> pathlib.Path:
    """
    Initializes the tensorflow cache dir.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Cache dir path.

    Returns
    -------
    pathlib.Path
        Cache dir path.
    """
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    print(f"Cache directory: {cache_dir}")
    return cache_dir
