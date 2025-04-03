##  <a name='TableofContents'></a> Table of Contents
<!-- vscode-markdown-toc -->
1. [Getting Started](#GettingStarted)
	* 1.1. [Reconstruct a phantom raw data set using the MRD client/server pair](#ReconstructaphantomrawdatasetusingtheMRDclientserverpair)
	* 1.2. [Creating a custom reconstruction/analysis module](#Creatingacustomreconstructionanalysismodule)
		* 1.2.1. [Adding a raw k-space filter](#Addingarawk-spacefilter)
		* 1.2.2. [Adding an image processing filter](#Addinganimageprocessingfilter)
	* 1.3. [Using raw data from an MRI scanner](#UsingrawdatafromanMRIscanner)
	* 1.4. [Using DICOM images as input data](#UsingDICOMimagesasinputdata)
2. [Setting up a working environment for the Python MRD client/server](#SettingupaworkingenvironmentforthePythonMRDclientserver)
	* 2.1. [Setting up a devcontainer environment](#Settingupadevcontainerenvironment)
	* 2.2. [Setting up a conda environment](#Settingupacondaenvironment)
3. [Spiral Reconstruction](#SpiralReconstruction)
     * 3.1 [Spiral View-Sharing Reconstruction](#SpiralVS)
     * 3.2 [Spiral Temporal Finite Difference (TFD) Reconstruction](#SpiralTFD)
     * 3.3 [Spiral Dual Temporally Constrained Reconstruction](#SpiralXDGRASP)
4. [Code design](#Codedesign)
5. [Saving incoming data](#Savingincomingdata)
6. [Startup scripts](#Startupscripts)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->
##  1. <a name='GettingStarted'></a>Getting Started
###  1.1. <a name='ReconstructaphantomrawdatasetusingtheMRDclientserverpair'></a>Reconstruct a phantom raw data set using the MRD client/server pair

1. Set up a working environment for the MRD client/server pair using [conda](#Settingupacondaenvironment) or [Docker](#SettingupaDockerenvironment).

1. In a command prompt, generate a sample raw dataset:
    ```
    python generate_cartesian_shepp_logan_dataset.py -o phantom_raw.h5
    ```

    For a Docker environment, this should be performed in the same [Docker container as the client](#Dockerclient).

    MRD data is stored in the HDF file format in a hierarchical structure in groups.  The above example creates a ``dataset`` group containing:
    ```
    /dataset/data   Raw k-space data
    /dataset/xml    MRD header
    ```

1. Start the server in verbose mode by running:
    ```
    python main.py -v
    ```

    For a Docker environment, this should be performed in a separate [Docker container](#Dockerserver).

1. Start the client and send data to the server for reconstruction:
    ```
    python client.py -G dataset -o phantom_img.h5 phantom_raw.h5
    ```
    The ``-G`` argument specifies the group name in the output file, the ``-o`` argument specifies the output file, and the last argument is the input file.

    If using [conda](#Settingupacondaenvironment), this should be run in a new command prompt.  For Docker, this can be run in the same container from step 1.

    MRD image data are also stored in HDF files arranged by groups. If the ``-G`` argument is not provided, a group name will be automatically created with the current date and time.  This may be useful when running the client multiple times, as multiple groups, organized by date/time, can be stored in the same output file.  Images are further grouped by series index, with a sub-group named ``image_x``, where x is image_series_index in the ImageHeader.  For example:
    ```
    /dataset/image_0/data         Image data
    /dataset/image_0/header       MRD ImageHeader structure
    /dataset/image_0/attributes   MRD MetaAttributes text
    ```

1. The [mrd2gif.py](mrd2gif.py) program can be used to convert an MRD Image file into an animated GIF for quick previewing:
    ```
    python mrd2gif.py phantom_img.h5
    ```
    A GIF file (animated if multiple images present) is generated in the same folder as the MRD file with the same base file name and the group and sub-groups appended.

    The reconstructed images in /tmp/phantom_raw.h5 can be opened in any HDF viewer such as https://www.hdfgroup.org/downloads/hdfview/.  In Python, the [ismrmrd-python-tools](https://github.com/ismrmrd/ismrmrd-python-tools) repository has an interactive ``imageviewer`` tool that can be used to view MRD formatted HDF files.  The syntax is:
    ```
    python imageviewer.py phantom_img.h5
    ```

    In MATLAB, the [ISMRMRD](https://github.com/ismrmrd/ismrmrd) library contains helper classes to load and view MRD files.  The files can also be read using MATLAB’s built-in HDF functions:
    ```
    img = h5read('/tmp/phantom_img.h5', '/dataset/image_0/data');
    figure, imagesc(img), axis image, colormap(gray)
    ```

###  1.2. <a name='Creatingacustomreconstructionanalysismodule'></a>Creating a custom reconstruction/analysis module
The MRD server has a modular design to allow for easy integration of custom reconstruction or image analysis code.  The config that is passed by the client (e.g. the `--config` (`-c`) argument in [client.py](client.py)) is interpreted by the server as the "module" that should be executed to parse the incoming data.  For example, a config of `invertcontrast` will select [invertcontrast.py](invertcontrast.py) as the module to be run.  Additional modules can be added simply be creating the appropriately named .py file in the Python path (e.g. the current folder).

If a file/module corresponding to the selecting config cannot be found, the server will fall back to a default config.  The default config can be provided to [main.py](main.py) using the `--defaultConfig` (`-d`) argument.  It is recommended that the default config argument be set in the `CMD` line of the Dockerfile when building an image to indicate the intended config to be run.

####  1.2.1. <a name='Addingarawk-spacefilter'></a>Adding a raw k-space filter
In this example, a Hanning filter is applied to raw k-space data.

1. Create a copy of [invertcontrast.py](invertcontrast.py) named ``filterkspace.py``.  For other workflows, it may be preferable to start with another example such as [simplefft.py](simplefft.py).

1. The NumPy library contains a [Hanning filter](https://numpy.org/doc/stable/reference/generated/numpy.hanning.html) that can be applied to k-space data before the Fourier transform.

1. In the ``process_raw()`` function the ``filterkspace.py`` file, find the [section where raw k-space data is already sorted into a Cartesian grid, just prior to Fourier transform](https://github.com/usc-mrel/python-ismrmrd-server/blob/27454bd9f1a2c7fd3928bfa0767840b0d015d988/invertcontrast.py#L177).  Add a Hanning filter and perform element-wise multiplication of the k-space data with the filter:

    **Old code:**
    ```
    # Fourier Transform
    data = fft.fftshift( data, axes=(1, 2))
    data = fft.ifft2(    data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))
    ```

    **New code:**
    ```
    # Apply Hanning filter
    logging.info("Applying Hanning filter to k-space data")
    filt = np.sqrt(np.outer(np.hanning(data.shape[1]), np.hanning(data.shape[2])))
    filt = np.expand_dims(filt, axis=(0,3))
    data = np.multiply(data,filt)
    np.save(debugFolder + "/" + "rawFilt.npy", data)

    # Fourier Transform
    data = fft.fftshift( data, axes=(1, 2))
    data = fft.ifft2(    data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))
    ```

1. The module used by the server is specified by the ``config`` option on the client side.  The Server class used in this code [attempts to find](https://github.com/usc-mrel/python-ismrmrd-server/blob/6684b4d17c0591e64b34bc06fdd06d78a2d8c659/server.py#L105) a Python module matching the name of the config file if it doesn't match one of the default examples.  [Start the server](https://github.com/usc-mrel/python-ismrmrd-server#ReconstructaphantomrawdatasetusingtheMRDclientserverpair) and in a separate window, run the client with the ``-c filterkspace`` option to specify the new config:
    ```
    python client.py -c filterkspace -o phantom_img.h5 phantom_raw.h5
    ```

1. Create a GIF preview of the filtered image:
    ```
    python mrd2gif.py phantom_img.h5
    ```

####  1.2.2. <a name='Addinganimageprocessingfilter'></a>Adding an image processing filter
In this example, a high-pass filter is applied to images.

1. Create a copy of [invertcontrast.py](invertcontrast.py) named ``filterimage.py``.  For other workflows, it may be preferable to start with another example such as [analyzeflow.py](analyzeflow.py).

1. The Python [Pillow](https://github.com/python-pillow/Pillow/) library contains a high-pass filter named [FIND_EDGES](https://pythontic.com/image-processing/pillow/edge-detection).  Add the following line to the top of the newly created ``filterimage.py`` to import this library:
    ```
    from PIL import Image, ImageFilter
    ```

1. In the ``process_image()`` function the ``filterimage.py`` file, find the sections where the incoming images are being [normalized](https://github.com/usc-mrel/python-ismrmrd-server/blob/6684b4d17c0591e64b34bc06fdd06d78a2d8c659/invertcontrast.py#L261) and [filtered](https://github.com/usc-mrel/python-ismrmrd-server/blob/6684b4d17c0591e64b34bc06fdd06d78a2d8c659/invertcontrast.py#L267). The Pillow image filters require images with values in the range 0-255, so replace these two sections as follows:

    **Old code:**
    ```
    # Normalize and convert to int16
    data = data.astype(np.float64)
    data *= maxVal/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Invert image contrast
    data = maxVal-data
    data = np.abs(data)
    data = data.astype(np.int16)
    np.save(debugFolder + "/" + "imgInverted.npy", data)
    ```

    **New code:**
    ```
    # Normalize to range 0-255
    data = data.astype(np.float64)
    data *= 255/data.max()

    # Apply a 2D high-pass filter for each image
    logging.info("Applying high-pass image filter")
    from PIL import Image, ImageFilter
    for iImg in range(data.shape[-1]):
        im = Image.fromarray(np.squeeze(data[...,iImg])).convert('RGB')
        im = im.filter(ImageFilter.FIND_EDGES)
        data[:,:,0,0,iImg] = np.asarray(im)[...,0]

    # Rescale back to 16-bit
    data = data * maxVal/data.max()
    data = data.astype(np.int16)
    np.save(debugFolder + "/" + "imgFiltered.npy", data)
    ```

1. The module used by the server is specified by the ``config`` option on the client side.  The Server class used in this code [attempts to find](https://github.com/usc-mrel/python-ismrmrd-server/blob/6684b4d17c0591e64b34bc06fdd06d78a2d8c659/server.py#L105) a Python module matching the name of the config file if it doesn't match one of the default examples.  [Start the server](https://github.com/usc-mrel/python-ismrmrd-server#ReconstructaphantomrawdatasetusingtheMRDclientserverpair) and in a separate window, run the client with the ``-c filterimage`` option to specify the new config:
    ```
    python client.py -c filterimage -o phantom_img.h5 phantom_raw.h5
    ```

1. Create a GIF preview of the filtered image:
    ```
    python mrd2gif.py phantom_img.h5
    ```

###  1.3. <a name='UsingrawdatafromanMRIscanner'></a>Using raw data from an MRI scanner
Raw data from MRI scanners can be converted into MRD format using publicly available conversion tools such as [siemens_to_ismrmrd](https://github.com/ismrmrd/siemens_to_ismrmrd), [ge_to_ismrmrd](https://github.com/ismrmrd/ge_to_ismrmrd), [philips_to_ismrmrd](https://github.com/ismrmrd/philips_to_ismrmrd), and [bruker_to_ismrmrd](https://github.com/ismrmrd/bruker_to_ismrmrd).  These can be used as input data for the client as part of streaming MRD framework.

For Siemens data, raw data in .dat format can be converted and processed as follows:
1. Convert a raw data file named ``gre.dat`` into MRD format:
    ```
    siemens_to_ismrmrd -Z -f gre.dat -o gre_raw.h5
    ```

    The following options are used:
    ```
    -Z  Convert all acquisitions from a multi-measurement (multi-RAID) file
    -f  Input file (Siemens raw data .dat file)
    -o  Output file (MRD raw data .h5 file)
    ```

    If the input file is a multi-RAID file, then several output files are created such as ``gre_raw_1.h5`` and ``gre_raw_2.h5``.  The first measurements are dependency data while the main acquisition is in the last numbered file.

1. [Start the server](https://github.com/usc-mrel/python-ismrmrd-server#ReconstructaphantomrawdatasetusingtheMRDclientserverpair) and in a separate window, run the client using the converted file:
    ```
    python client.py -c invertcontrast -o gre_img.h5 gre_raw_2.h5
    ```

    Note that the invertcontrast example module only does basic Fourier transform reconstruction and does not support undersampling or more complex acquisitions.

###  1.4. <a name='UsingDICOMimagesasinputdata'></a>Using DICOM images as input data
For image processing workflows, DICOM images can be used as input by converting them into MRD format.  The [dicom2mrd.py](dicom2mrd.py) script can be used to convert DICOMs to MRD, while [mrd2dicom.py](mrd2dicom.py) can be used to perform the inverse.

1. Create a folder containing DICOM files with file extensions .ima or .dcm.  Files can also be organized in sub-folders if desired.

1. Run the dicom2mrd conversion script:
    ```
    python dicom2mrd.py -o dicom_img.h5 dicoms
    ```
    Where the DICOM files are in a folder called ``dicoms`` and an output file ``dicom_img.h5`` is created containing the MRD formatted images.

1. [Start the server](https://github.com/usc-mrel/python-ismrmrd-server#ReconstructaphantomrawdatasetusingtheMRDclientserverpair) and in a separate window, run the client using the converted file:
    ```
    python client.py -c invertcontrast -o dicom_img_inverted.h5 dicom_img.h5
    ```

1. Convert the output MRD file back to a folder of DICOMs:
    ```
    python mrd2dicom.py dicom_img_inverted.h5
    ```

##  2. <a name='SettingupaworkingenvironmentforthePythonMRDclientserver'></a>Setting up a working environment for the Python MRD client/server
###  2.1. <a name='Settingupadevcontainerenvironment'></a>Setting up a devcontainer environment
[Development containers (devcontainers)](https://code.visualstudio.com/docs/devcontainers/containers) are a convenient way of using a Docker image as a working environment instead of installing packages locally.  If the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension is installed in Visual Studio Code, then a prompt will appear when opening this folder to re-open it in a devcontainer.  The devcontainer is also compatible with [GitHub Codespaces](https://github.com/features/codespaces).  Further details can be found in [doc/devcontainers.md](doc/devcontainers.md).

###  2.2. <a name='Settingupacondaenvironment'></a>Setting up a conda environment
Conda is a Python environment manager that is useful for creating and maintaining Python packages and their dependencies.  It is available either as part of the larger [Anaconda](https://www.anaconda.com/) product, or separately as part of [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  Although not required, it's helpful in setting up an environment for the Python ISMRMD client/server.  [Mamba](https://mamba.readthedocs.io/en/latest/) and [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) are drop-in replacements for Conda that are often faster at resolving dependencies.  The following instructions are for micromamba, but the command `micromamba` can be replaced with `conda` if conda is preferred.

1. Download and install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for your operating system.

1. Download and install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your operating system.

1. For Windows, open a powershell prompt.  In MacOS and Linux, open a standard command prompt.

1. Clone (download) this repository
    ```
    git clone https://github.com/usc-mrel/python-ismrmrd-server.git
    ```

1. Change into this respository's directory and create a new conda environment for MRD using the dependencies listed in [environment.yml](environment.yml).  If using Windows, then [environment_windows.yml](environment_windows.yml) should be used instead.
    ```
    cd python-ismrmrd-server
    micromamba create -f environment.yml
    ```

1. Active the new MRD environment
    ```
    micromamba activate mrd
    ```

1. If using Windows, then install the ISMRMRD Python library through pip.  For MacOS and Linux, this was installed through conda.
    ```
    pip install ismrmrd
    ```

To use this environment in the future, open a command prompt and run ``micromamba activate mrd``.

##  3. <a name='SpiralReconstruction'></a>Spiral Reconstruction

Spiral reconstruction can reconstruct 2D spiral sequences generated by [RTSpiral PyPulseq](https://github.com/usc-mrel/rtspiral_pypulseq). All three reconstructions have the following prerequisites:
1. **The metafile:** This is a ".mat" file generated by the Spiral design script that contains necessary information to perform reconstruction that is not available in the raw data. The filename is the sequence signature provided by Pulseq. This includes the trajectory, DCF, designed FoV and resolution, and Golden Angle parameters. Default searchpath is `seq_meta/` directory under the current working directory.
2. **The config file:** This is a ".toml" file that sets the reconstruction parameters. This file will be read every time a new recon has arrived, meaning we don't have to restart the recon server every time there is a config change. File paths and keys for each reconstruction will be given below.

Also, all three reconstruction have GIRF correction available, however, GIRF measurements for the target scanner has to be put at as a ".mat" file under the project directory, and the search path for the file needs to be edited in `GIRF.py`.


### 3.1 <a name='SpiralVS'></a>Spiral View-Sharing Reconstruction

This reconstruction is for real-time view-sharing reconstruction. It will be run if the config name `simplenufft1arm` sent from the client. On a reconstruction server with a sufficient GPU, it should be able to catch up with the acquisition. It uses [SigPy](https://github.com/mikgroup/sigpy) NUFFT for the reconstruction. 
Configuration is done via `${WORKDIR}/configs/rtspiral_vs_config.toml`. Available parameters are as follows:

```toml
"metafile_paths" = ["/PATH/TO/METAFILES"] # This tells the recon where to look for the metafiles (HASH.mat). Default is ${WORKDIR}/seq_meta

[reconstruction]
"apply_girf"        = true
"arms_per_frame"    = 55	# How many arms to grid per frame (temporal footprint)
"window_shift"      = 17	# How many arms per frame will be shifted.
"gpu_device"        = 2		# GPU ID to be used.
"coil_combine"      = "adaptive"	# "adaptive" or "rssq". Adaptive uses Inati's method to calculate coil maps from the first frames. "rssq" uses root sum of squares.
"save_complex"      = false # Save complex float images, rather than scaled magnitude int images.
"ignore_arms_per_frame" = false # When true, arms per frame and window shift are ignored and arms to grid will be interleaves parameter in the metadata. Useful for linear ordering.

```

### 3.2 <a name='SpiralTFD'></a>Spiral Temporal Finite Difference (TFD) Reconstruction

This is an iterative temporal finite difference (TFD) reconstruction. It will be run if the config name `rtspiral_bart_tvrecon` sent from the client. It uses BART's `nlinv` to calculate coil maps and `pics` to perform iterative reconstruction. It can divide data into chunks if it does not fit into a GPU.
Configuration is done via `${WORKDIR}/configs/rtspiral_bart_tvrecon_config.toml`. Available parameters are as follows:

```toml
[reconstruction]
"apply_girf"        = true
"gpu_device"        = 0 	# GPU ID to be used

"arms_per_frame"    = 7		# How many arms to grid per frame (temporal footprint)
"num_recon_frames"  = 2000	# Number of frames to be reconstructed (extra arms at the end will be discarded).
"num_chunks"        = 5         # Divide time frames into chunks so that they fit into a single GPU.
"fov_oversampling"  = 1.5       # Oversampling factor for FOV
"reg_lambda"        = 0.0002	# Temporal regularizer.
"num_iter"          = 120	# Number of iterations for the TFD recon.

"BART_TOOLBOX_PATH" = "/PATH/TO/BART/TOOLBOX"
```

### 3.3 <a name='SpiralXDGRASP'></a> Spiral Dual Temporally Constrained Reconstruction

This is an dual temporally constrained reconstruction, similar to XDGRASP. It will be run if the config name `spiral_xdgrasp_recon` sent from the client. It uses BART's `nlinv` to calculate coil maps and `pics` to perform iterative reconstruction. It needs cardiac and respiratory navigators to run. It only accepts pilot tone (**PT**) for the respiratory navigator. It accepts ECG or PT for the cardiac navigator. It assumes PT has the waveform ID 1025 in the MRD.
Configuration is done via `${WORKDIR}/configs/spiral_xdgrasp_config.toml`. Available parameters are as follows:

```
[reconstruction]
"apply_girf"         = true
"arm_discard_pre"    = 5 	# [s] First arm_discard second will be excluded from the reconstruction.
"arm_discard_post"   = 5 	# [s] Last arm_discard second will be excluded from the reconstruction.
"reg_lambda_resp"    = 0.0005	# Regularizer along respiratory dimension
"reg_lambda_card"    = 0.0005	# Regularizer along the cardiac dimension
"num_iter"           = 300	# Number of optimization iterations
"fov_oversampling"   = 1        # Oversampling factor for FOV
"gpu_device"         = 3	# GPU ID to be used for the reconstruction
"BART_TOOLBOX_PATH"  = "/PATH/TO/BART/TOOLBOX"

[binning]
"n_respiratory_bins" = 12	# Number of respiratory bins
"n_cardiac_bins"     = 20	# Number of cardiac bins
"trigger_source"     = "pilottone_derivative" # "ecg", "pilottone", "pilottone_derivative". This is the trigger source for the cardiac only. Respiratory always uses PT. 
"ecg_pilottone_delay"= 0.15 	# [s] Delay to be compensated for pilottone in order to match cardiac phases with ECG.
"resp_discard"       = 0.1 	# Ratio of the extreme states to be discarded for the respiratory waveform
"fill_small_bins"    = true 	# When true, bins will be zero filled to the maximum number of arms. If false, they will be cropped to the minimum.
```

##  4. <a name='Codedesign'></a>Code design
This code is designed to provide a reference implementation of an MRD client/server pair.  It is modular and can be easily extended to include additional reconstruction/analysis programs.

- [main.py](main.py):  This is the main program, parsing input arguments and starting a "Server" class instance.

- [server.py](server.py):  The "Server" class determines which reconstruction algorithm (e.g. "simplefft" or "invertcontrast") is used for the incoming data, based on the requested config information.

- [connection.py](connection.py): The "Connection" class handles network communications to/from the client, parsing streaming messages of different types, as detailed in the [MRD documentation](https://ismrmrd.readthedocs.io/en/latest/mrd_messages.html).  The connection class also saves incoming data to MRD files if this option is selected.

- [constants.py](constants.py): This file contains constants that define the message types of the MRD streaming data format.

- [mrdhelper.py](mrdhelper.py): This class contains helper functions for commonly used MRD tasks such as copying header information from raw data to image data and working with image metadata.

- [client.py](client.py): This script can be used to function as the client for an MRD streaming session, sending data from a file to a server and saving the received images to a different file.  Additional description of its usage is provided below.

- [generate_cartesian_shepp_logan_dataset.py](generate_cartesian_shepp_logan_dataset.py): Creates an MRD raw data file of a Shepp-Logan phantom with Cartesian sampling.  Borrowed from the [ismrmrd-python-tools](https://github.com/ismrmrd/ismrmrd-python-tools) repository.

- [dicom2mrd.py](dicom2mrd.py): This program converts a folder of DICOM images to an MRD image .h5 file, allowing DICOM images to be use as input for MRD streaming data.

- [mrd2dicom.py](mrd2dicom.py): This program converts an MRD image .h5 file to a folder of DICOM images.

- [mrd2gif.py](mrd2gif.py): This program converts an MRD image .h5 file into an animated GIF for quick previews.

There are several example "modules" that can be selected by specifying their name via the config (`-c`) argument:
- [invertcontrast.py](invertcontrast.py): This program accepts both incoming raw data as well as image data.  The image contrast is inverted and images are sent back to the client.

- [simplefft.py](simplefft.py): This file contains code for performing a rudimentary image reconstruction from raw data, consisting of a Fourier transform, sum-of-squares coil combination, signal intensity normalization, and removal of phase oversampling.

- [rgb.py](rgb.py): This program accepts incoming image data, applies a jet colormap, and sends RGB images back to the client.

- [analyzeflow.py](analyzeflow.py): This program accepts velocity phase contrast image data and performs basic masking.

##  5. <a name='Savingincomingdata'></a>Saving incoming data
It may be desirable for the MRD server to save a copy of incoming data from the client.  For example, if the client is an MRI scanner, then the saved data can be used for offline simulations at a later time.  This may be particularly useful when the MRI scanner client is sending image data, as images are not stored in a scanner's raw data file and would otherwise require offline simulation of the MRI scanner reconstruction as well.

The feature may be turned on by starting the server with the ``-s`` option (disabled by default).  Data files are named by the current date/time and stored in ``/tmp/share/saved_data``.  The saved data folder can be changed using the ``-S`` option.  For example, to turn on saving of incoming data in the ``/tmp`` folder, start the server with:
```
python main.py -s -S /tmp
```

The ``-s`` flag is enabled in the startup script [start-fire-python-server-with-data-storage.sh](start-fire-python-server-with-data-storage.sh).

Alternatively, this feature can be enabled on a per-session basis when the client calls the server with the config ``savedataonly``.  In this mode, incoming data (raw or image) is saved, but no processing is done and no images are sent back to the client.

The resulting saved data files are in MRD .h5 format and can be used as input for ``client.py`` as detailed above.

##  6. <a name='Startupscripts'></a>Startup scripts
There are three scripts that may be useful when starting the Python server in a chroot environment (i.e. on the scanner).  When using this server with FIRE, the startup script is specified in the fire.ini file as ``chroot_command``.  The scripts are:

- [start-fire-python-server.sh](start-fire-python-server.sh):  This script takes one optional argument, which is the location of a log file.  If not provided, logging outputs are discarded.

- [sync-code-and-start-fire-python-server.sh](sync-code-and-start-fire-python-server.sh):  This script copies all files from ``/tmp/share/code/`` to ``/opt/code/python-ismrmrd-server/``.  These paths are relative to the chroot container and when run with FIRE, ``/tmp/share/code/`` is a shared folder from the host computer at ``%CustomerIceProgs%\fire\share\code\``.  This "sync" step allows Python code to be modified on the host computer and executed by the Python reconstruction process.  The Python reconstruction program is then started in the same way as in [start-fire-python-server.sh](start-fire-python-server.sh).  This is the startup script specified in the default fire.ini configuration file.  However, this script should not be used in stable projects, as overwriting existing files with those in ``/tmp/share/code`` is likely undesirable.

- [start-fire-python-server-with-data-storage.sh](start-fire-python-server-with-data-storage.sh):  This script is the same as [start-fire-python-server.sh](start-fire-python-server.sh), but saves incoming data (raw k-space readouts or images) to files in ``/tmp/share/saved_data``, which is on the host computer at ``%CustomerIceProgs%\fire\share\saved_data``.  This is useful to save a copy of MRD formatted data (particularly for image-based workflows) for offline development, but this option should be used carefully, as raw data can be quite large and can overwhelm the limited hard drive space on the Windows host.
