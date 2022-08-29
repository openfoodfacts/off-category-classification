ARG TF_VERSION=2.8.2-jupyter

from tensorflow/tensorflow:$TF_VERSION

ARG USER_UID=1000
ARG USER_GID=1000

# create user and app dir
RUN groupadd --gid $USER_GID tf && \
    useradd --create-home --shell /bin/bash -u $USER_UID --gid tf tf && \
    mkdir /app && \
    chown tf:tf /app

WORKDIR /app

COPY --chown=tf:tf ./ /app

# we remove already installed products from requirements
RUN  pip install -U pip && \
    cat requirements.txt requirements-dev.txt|grep -v '\(numpy\|tensorflow\|keras\|tensorboard\|robotoff\|ipython\|notebook\)==' > requirements-stripped.txt && \
    pip install -r requirements-stripped.txt

USER  tf
ENTRYPOINT ["bash"]
CMD ["-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/app/experiments --no-browser --ip 0.0.0.0"]
