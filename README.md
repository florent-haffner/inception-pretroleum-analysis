# Inception Petroleum Analysis
This repository shows the code used in "Inception for Petroleum Analysis" for cetane number prediction. Study published in Fuel (Elsevier), 2025. 

## The IPA Deep CNN architecture

![The IPA architecture](docs/images/Architecture_IPA___PlotNeuralNet.jpg)

Convolutions with kernel size 3 are shown in blue, convolutions with kernel size 1 are shown in red. 
The numbers of neurons are shown under each computational blocks. 
For more details, verify on the paper under the section -> Methodology / Deep learning. 

## Getting started

### Matlab
Install [MATLAB R2022b](https://www.mathworks.com/products/new_products/release2022b.html) and [Eigenvector PLS_Toolbox version 9.2](https://wiki.eigenvector.com/index.php?title=Release_Notes_Version_9_2).

## Launching environment
### Matlab

Open the script within the eponymous software.

### Python
The Python environment uses Python 3.13 and TensorFlow 2.21 (Keras 3), the latest stable releases as of November 2026. 
All environment files live in `env/` (`requirements.txt`, `environment.yml`, `Dockerfile`) and every command goes through the `Makefile`: 

```bash
make help
```

#### Docker (CPU)
Build the image, then run it. Trained weights are written to `./model`:
```bash
make docker-build
```
```bash
make docker-run
```

#### Local virtualenv
Creates `.venv` (with [uv](https://docs.astral.sh/uv/) if installed, otherwise `python3.13 -m venv`) and launches models calibration:
```bash
make run
```

#### Conda
This environment expects miniconda/anaconda already installed.
```bash
make conda-env
```
```bash
make run-conda
```

#### GPU
On Linux, replace `tensorflow` with `tensorflow[and-cuda]` in `env/requirements.txt` (CUDA 12 and cuDNN 9 are installed by pip).

#### Customising the pipeline
The script `scripts/models-calibration.py` can be adapted to different needs.

Inside the main() function, three different approaches are possible:
- comparing the two models using the compare_models() function,
- using a very simple approach via the basic_usage() function,
- modifying a configuration via config_from_dict().

This code has been rewritten to be more modular and allow for testing different model configurations, as well as extension to other architectures.
To do this, a practitioner can develop the following classes (found on eponym file under directory "src.model_utils"): ModelType, BaseModelConfig, BaseModelPipeline, and finally update class ModelFactory (especially methods: _registry and _config_registry. 


### Notes about our data

![Data mangement](docs/images/data_mgt.drawio.png)

Information : all y are already normalized to follow entreprise policy, the scaler is not given. Therefore, calculated metrics will be roughly 10 times smaller than on the publication. 
- PLS will be fed using X and y from 'data/raw-kennard-full_range.mat', y is already normalized (centered) but X are raw.
- IPA and DeepSpectra will be fed using y from 'y_train_scaled.csv' and 'y_test_scaled.csv' (mean 0 & std 1), the X comes raw from 'data/raw-kennard-full_range.mat' or feature selected from 'data/raw-kennard-reduced_range.mat'

