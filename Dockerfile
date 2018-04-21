FROM continuumio/anaconda3:latest

RUN /opt/conda/bin/conda update -y -n base conda

RUN /opt/conda/bin/conda install -y --quiet jupyter
RUN /opt/conda/bin/conda install -c conda-forge -y --quiet xgboost
RUN /opt/conda/bin/conda install -y --quiet tensorflow
RUN /opt/conda/bin/conda install -y --quiet keras

RUN useradd -m myuser
USER myuser

WORKDIR /home/myuser

COPY ./notebook notebook
COPY ./input input

WORKDIR /home/myuser/notebook

EXPOSE 8888

CMD /opt/conda/bin/jupyter notebook run.py
#CMD bash