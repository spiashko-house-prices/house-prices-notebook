FROM continuumio/anaconda3:latest

RUN /opt/conda/bin/conda update -y -n base conda

RUN /opt/conda/bin/conda install -y --quiet jupyter
RUN /opt/conda/bin/conda install -c conda-forge -y --quiet xgboost
RUN /opt/conda/bin/conda install -y --quiet tensorflow
RUN /opt/conda/bin/conda install -y --quiet keras

RUN useradd -m myuser

WORKDIR /home/myuser

COPY ./notebook/data_analise_and_learning.ipynb notebook/data_analise_and_learning.ipynb
COPY ./input input

RUN /opt/conda/bin/jupyter trust notebook/data_analise_and_learning.ipynb

RUN chown -R myuser:myuser /home/myuser/notebook

WORKDIR /home/myuser/notebook

USER myuser

RUN /opt/conda/bin/jupyter notebook --generate-config

COPY jupyter_notebook_config.conf /home/myuser/.jupyter/jupyter_notebook_config.py

EXPOSE $PORT

CMD /opt/conda/bin/jupyter notebook
#CMD bash