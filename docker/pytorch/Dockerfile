# Start from the fire-python image that has all base dependencies
FROM kspacekelvin/fire-python

# Install PyTorch and its dependencies
COPY python-ismrmrd-server/docker/pytorch/requirements.txt /
RUN apt-get install -y git && \
    apt-get install -y libglib2.0-0 && \
    apt install libgl1-mesa-glx -y  && \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    rm -r /root/.cache/pip

# # If building from the GitHub repo, uncomment the below section, open a command
# # prompt in the folder containing this Dockerfile and run the command:
# #    docker build --no-cache -t kspacekelvin/fire-python ./
# RUN  cd /opt/code \
#      && git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

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

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log"]