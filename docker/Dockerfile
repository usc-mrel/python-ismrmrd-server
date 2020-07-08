FROM python:3.8.2-alpine3.11 AS mrd_converter

RUN     apk --no-cache add hdf5 hdf5-dev --repository http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
     && apk --no-cache add cmake build-base boost-dev boost-static libxml2-dev libxslt-dev git fftw-dev

RUN  mkdir -p /opt/code

# ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git

RUN cd /opt/code/siemens_to_ismrmrd && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# ----- Start another clean build without all of the build dependencies of siemens_to_ismrmrd -----
FROM python:3.8.2-alpine3.11
LABEL maintainer="Kelvin Chow (kelvin.chow@siemens.com)"

# Copy siemens_to_ismrmrd from last stage and re-add necessary dependencies
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd /usr/local/bin/siemens_to_ismrmrd
COPY --from=mrd_converter /usr/local/lib/libismrmrd*        /usr/local/lib/
RUN     apk --no-cache add hdf5 --repository http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
     && apk --no-cache add libxslt

# Dependencies for Python MRD server
RUN     apk --no-cache add hdf5 hdf5-dev --repository http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
     && apk --no-cache add bash git openssh gcc libc-dev \
     && pip install --no-cache-dir pyxb h5py

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python.git \
     && cd /opt/code/ismrmrd-python \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python-tools.git \
     && cd /opt/code/ismrmrd-python-tools \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

# Cleanup files not required after compiling
RUN  apk del gcc git
RUN  rm -r /root/.cache/pip

CMD [ "/usr/local/bin/python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]
