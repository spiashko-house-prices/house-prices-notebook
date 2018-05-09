FROM continuumio/anaconda3:latest

RUN /opt/conda/bin/conda update -y -n base conda

RUN /opt/conda/bin/conda install -y --quiet jupyter
RUN /opt/conda/bin/conda install -c conda-forge -y --quiet xgboost
RUN /opt/conda/bin/conda install -y --quiet tensorflow
RUN /opt/conda/bin/conda install -y --quiet keras
RUN /opt/conda/bin/conda install -y --quiet pymongo

RUN useradd -m myuser

WORKDIR /home/myuser

COPY ./notebook/data_analise.ipynb notebook/data_analise.ipynb
COPY ./notebook/learning.ipynb notebook/learning.ipynb
RUN mkdir /home/myuser/.jupyter
COPY jupyter_notebook_config.conf /home/myuser/.jupyter/jupyter_notebook_config.py

WORKDIR /home/myuser/
RUN /opt/conda/bin/jupyter trust notebook/data_analise.ipynb
RUN /opt/conda/bin/jupyter trust notebook/learning.ipynb

RUN chown -R myuser:myuser /home/myuser

WORKDIR /home/myuser/.jupyter

USER myuser

EXPOSE $PORT

CMD /opt/conda/bin/jupyter notebook --ip='*' --notebook-dir=/home/myuser/notebook --port=$PORT --no-browser
#CMD bash