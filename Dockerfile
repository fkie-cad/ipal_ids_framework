FROM ubuntu:22.04

RUN apt-get update \
    && apt-get -y install software-properties-common sudo g++ \
    && apt-get -y install pip vim
RUN apt-get -y install libgsl-dev git

# create a user and add them to sudoers
RUN useradd -ms /bin/bash ipal && echo "ipal ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ipal
USER ipal

WORKDIR /home/ipal/ipal_ids_framework/
COPY --chown=ipal . .

# Install IIDS framework
WORKDIR /home/ipal/ipal_ids_framework/
RUN sudo pip install numpy # for ar to install correctly
RUN sudo pip install .
RUN sudo pip install -r requirements-dev.txt

CMD /bin/bash
