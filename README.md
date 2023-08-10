# CSCI-635-Group-4-ResNet

### Setup

To use the code in this repo, you will need to have PyTorch installed.
To set up PyTorch locally, go to the [PyTorch website](https://pytorch.org/get-started/locally/)
and follow the installation instructions there.
Choosing the most up-to-date CUDA version as your compute platform is recommend if you are going
to be running the code on a GPU.

You can install the remaining requirements using PIP.
```
pip install -r requirements.txt
```

### Training Visualization

In order to view run metrics, before training a model be sure to start the 
[LabML server](https://github.com/labmlai/labml).
The local server can be easily started using the appropriate script.

Linux: [start_labml_server.sh](start_labml_server.sh)

Windows: [start_labml_server.bat](start_labml_server.bat)

Windows using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to run the server: [WSL_start_labml_server.bat](WSL_start_labml_server.bat)

### Training a ResNet Model

Training configurations can be changed in the main() method in [main.py](main.py?plain=1#L4).
Training can be done on either a CPU or a GPU.
The device to use is selected automatically - if a CUDA compatible GPU is available it will be used, 
otherwise training will default to using your CPU.  

Once you have started the LabML server, to begin training simply run [main.py](main.py).
```
python main.py
```
