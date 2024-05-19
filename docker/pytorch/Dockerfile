# ----- 1. Create devcontainer image -----
# Start from the fire-python devcontainer image that has all base dependencies
# Note: This does not contain the actual code for python-ismrmrd-server!
FROM kspacekelvin/fire-python-devcon AS python-mrd-pytorch-devcontainer

# Install PyTorch and its dependencies
RUN pip3 install --no-cache-dir torch torchvision \
    && rm -rf /root/.cache/pip

# ----- 2. Copy deployed code into the devcontainer for deployment -----
FROM python-mrd-pytorch-devcontainer as python-mrd-pytorch-runtime

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