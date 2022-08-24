from pathlib import Path
import os
import re
import shutil
import subprocess
import sys


# library not to install in colab
COLAB_EXCLUDE = {"tensorflow", "ipython", "notebook"}


def colab_requirements_iter():
    """Listed needed libs for colab notebooks"""
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
        if lib not in COLAB_EXCLUDE:
            yield requirement


def pip_install(requirements):
    result = subprocess.run(["pip", "install"] + requirements, capture_output=True)
    if result.returncode != 0:
        print(f"Error while running pip install {' '.join(requirements)}", file=sys.stderr)
        print(f"Output:{result.stdout.decode('utf-8')}")
        print(f"Errors:{result.stderr.decode('utf-8')}", file=sys.stderr)
        raise RuntimeError("pip install failed")
    print("Install successful")


def colab_pip_install():
    """Install needed parts for colab notebooks"""
    requirements = list(colab_requirements_iter())
    pip_install(requirements)


def init_git(branch_name=None, target_dir="experiments"):
    """Init git repo in colab"""
    initial = os.getcwd()
    # might be checked out here
    if os.path.exists("off-category-classification"):
        os.chdir("off-category-classification")
    inside_git = subprocess.run("git rev-parse --git-dir".split(), capture_output=True)
    if inside_git.returncode != 0:
        print("Cloning git repo")
        if os.path.exists("off-category-classification"):
            shutil.rmtree("off-category-classification")
        # init git repo and go in experiments
        result = subprocess.run("git clone https://github.com/openfoodfacts/off-category-classification.git".split(), capture_output=True)
        git_cloned = 1
        if result.returncode != 0:
            raise RuntimeError("Error on git clone", result.stderr.decode("utf-8"))
        os.chdir(f"{initial}/off-category-classification/")
    if branch_name:
        result = subprocess.run(f"git checkout {branch_name}".split(), capture_output=True) 
        if result.returncode != 0:
            print("Error on git checkout:\n", result.stderr.decode("utf-8"))
    if target_dir:
        os.chdir(f"{initial}/off-category-classification/{target_dir}")


def init_colab(branch_name=None):
    """Global init procedure for colab"""
    init_git(branch_name)
    import sys
    sys.path.append('..') # append a relative path to the top package to the search path
    # now import from real module to have right __file__
    from lib.colab import colab_pip_install
    print("Installing packages")
    colab_pip_install()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        branch_name = sys.argv[1]
    init_colab(branch_name=branch_name)