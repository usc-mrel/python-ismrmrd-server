# ----- 1. First stage to build ismrmrd and siemens_to_ismrmrd -----
FROM python:3.12.0-slim AS mrd_converter
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

RUN  apt-get update && apt-get install -y git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev
RUN  mkdir -p /opt/code

# ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout d364e03 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    git checkout v1.2.11 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# Create archive of ISMRMRD libraries (including symlinks) for second stage
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# ----- 2. Create a devcontainer without all of the build dependencies of MRD -----
FROM python:3.12.0-slim AS python_mrd_devcontainer

LABEL org.opencontainers.image.description="Python MRD Image Reconstruction and Analysis Server"
LABEL org.opencontainers.image.url="https://github.com/kspaceKelvin/python-ismrmrd-server"
LABEL org.opencontainers.image.authors="Kelvin Chow (kelvin.chow@siemens-healthineers.com)"

# Copy ISMRMRD files from last stage
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd from last stage
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd  /usr/local/bin/siemens_to_ismrmrd

# Add dependencies for siemens_to_ismrmrd
RUN apt-get update && apt-get install --no-install-recommends -y libxslt1.1 libhdf5-103 libboost-program-options1.74.0 libpugixml1v5 git dos2unix
RUN mkdir -p /opt/code

# Python MRD library
RUN pip3 install h5py==3.10.0 ismrmrd==1.14.0

RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd /opt/code/ismrmrd-python-tools && \
    pip3 install --no-cache-dir .

# matplotlib is used by rgb.py and provides various visualization tools including colormaps
# pydicom is used by dicom2mrd.py to parse DICOM data
RUN pip3 install --no-cache-dir matplotlib==3.8.2 pydicom==2.4.3 pynetdicom==2.0.2

# Cleanup files not required after installation
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip

# ----- 3. Copy deployed code into the devcontainer for deployment -----
FROM python_mrd_devcontainer AS python_mrd_runtime

# If building from the GitHub repo, uncomment the below section, open a command
# prompt in the folder containing this Dockerfile and run the command:
#    docker build --no-cache -t kspacekelvin/fire-python ./
# RUN cd /opt/code && \
#     git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

# If doing local development, use this section to copy local code into Docker
# image. From the folder containing the python-ismrmrd-server repo, uncomment
# the COPY line below and run the command:
#    docker build --no-cache -t fire-python-custom -f python-ismrmrd-server/docker/Dockerfile ./
COPY python-ismrmrd-server  /opt/code/python-ismrmrd-server

# Ensure startup scripts have Unix (LF) line endings, which may not be true
# if the git repo is cloned in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" | xargs dos2unix

# Ensure startup scripts are marked as executable, which may be lost if files
# are copied in Windows
RUN find /opt/code/python-ismrmrd-server -name "*.sh" -exec chmod +x {} \;

# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=invertcontrast"]

# Replace the above CMD with this ENTRYPOINT to allow allow "docker stop"
# commands to be passed to the server.  This is useful for deployments, but
# more annoying for development
# ENTRYPOINT [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]