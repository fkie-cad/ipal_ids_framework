FROM ipal-ids-base:v1

COPY . . 
RUN sudo pip install .
RUN sudo pip install -r requirements-dev.txt

CMD /bin/bash
