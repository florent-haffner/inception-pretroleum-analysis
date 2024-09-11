# Inception Petroleum Analysis
This repository shows the code used in the publication "Inception for Petroleum Analysis" for cetane number prediction. 

## The IPA Deep CNN architecture

![The IPA architecture](Architecture_IPA___PlotNeuralNet.jpg)

## Getting started
Information : y is already normalized (mean 0 & std 1), the scaler is not given. Therefore, calculated metrics will be 15 times smaller than on the publication. 

### Python
Follow the next commands:
```
conda create -n inception-petroleum-analysis python=3.10

conda activate inception-petroleum-analysis 

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

pip install "tensorflow<2.11"

pip install -r requirements.txt
```
### Matlab
Install [MATLAB R2022b](https://www.mathworks.com/products/new_products/release2022b.html) and [Eigenvector PLS_Toolbox version 9.2](https://wiki.eigenvector.com/index.php?title=Release_Notes_Version_9_2).

## Launching environment
### Matlab

Open the script within the eponymous software.

### Python

Activate the environment, then

#### IPA calibration
```
python -m scripts.IPA-calibration
```

#### DeepSpectra calibration
```
python -m scripts.DeepSpectra-calibration
```
