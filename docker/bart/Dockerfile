FROM kspacekelvin/fire-python

# Install BART.  The git package is still required to provide
# the Python wrapper
RUN apt-get update && apt-get install --no-install-recommends -y git bart
RUN cd /opt/code && \
    git clone https://github.com/mrirecon/bart.git
ENV PYTHONPATH=/opt/code/bart/python

# Cleanup files not required after installation
RUN  apt-get remove git -y          \
     && apt-get clean               \
     && rm -rf /var/lib/apt/lists/* \
     && rm -rf /root/.cache/pip
