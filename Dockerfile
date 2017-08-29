FROM python:2.7

WORKDIR /text_generation

ADD . /text_generation

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p /src && \
    chown keras /src


RUN pip install --upgrade pip && \
    pip install tensorflow && \
    pip install h5py && \
    git clone git://github.com/fchollet/keras.git&& \
    pip install git+git://github.com/fchollet/keras.git 
    
  
ENV NAME World

ENV PYTHONPATH='/src/:$PYTHONPATH'

CMD ["python","generate.py"] 



