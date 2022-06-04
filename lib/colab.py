from pathlib import Path
import re
import subprocess
import sys


try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False


# library not to install in collab
COLLAB_EXCLUDE = {"tensorflow", "ipython", "notebook"}


def colab_requirements_iter():
    """Listed needed libs for collab notebooks"""
    project_dir = Path(__file__).parent.parent
    candidates = list(filter(
        None,
      (l.split("#", 1)[0].strip() for l in open(project_dir / "requirements.txt"))
    ))
    candidates.extend(
        filter(
            None, (l.split("#", 1)[0].strip() for l in open(project_dir / "requirements-dev.txt"))
        )
    )
    requirements_sep = re.compile(r"([=>< ]+)")
    to_install = []
    for requirement in candidates:
        lib, *specifiers = requirements_sep.split(requirement, maxsplit=1)
        if lib not in COLLAB_EXCLUDE:
            yield requirement

def pip_install(requirements):
    result = subprocess.run(["pip", "install"] + requirements, capture_output=True)
    if result.returncode != 0:
        print(f"Error while running pip install {' '.join(requirements)}", file=sys.stderr)
        print(f"Output:{result.stdout.decode('utf-8')}")
        print(f"Errors:{result.stderr.decode('utf-8')}", file=sys.stderr)
        raise RuntimeError("pip install failed")
    return "Install successful"


def colab_pip_install(in_colab):
    """Install needed parts for collab notebooks"""
    if in_colab:
        requirements = list(colab_requirements_iter())
        return pip_install(requirements)

