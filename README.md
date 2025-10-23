# Simultaneous Temporal-Frequency-Variable Modeling for Power Forecasting
![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg?style=plastic)
![PyTorch 2.1.0](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![CUDA 11.8](https://img.shields.io/badge/cuda-11.8-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of SimTFV in the following paper: 
[Simultaneous Temporal-Frequency-Variable Modeling for Power Forecasting] (Manuscript submitted to IEEE Internet of Things Journal). The data preprocessing, hyperparameter settings, experimental setups (including ablation studies), training duration, hardware specifications, and inference latency can be found in the manuscript.

## Model Architecture

<p align="center">
<img src="./img/SimTFV.jpg" height = "520" width = "1064" alt="" align=center />
<br><br>
<b>Figure 1.</b> The architecture of our proposed SimTFV consists of three main components: (a) RevSTIN operation (right side), which mitigates the impact of
nonstationarity on the statistical characteristics of the input sequence and fuse the temporal-frequency features. An example that splits the input sequence into
four segments is presented. (b) TVA module, which efficiently extracts temporal-variable features in the encoder. (c) TCA module, which generates the output
feature maps in the decoder for producing the prediction results.
</p>


## Requirements
- python == 3.11.4
- numpy == 1.24.3
- pandas == 1.5.3
- scipy == 1.11.3
- torch == 2.1.0+cu118
- scikit-learn == 1.4.2
- thop

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Raw Data
ECL was acquired at: [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). Solar dataset was acquired at: [Solar](https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-). Wind was acquired at: [Wind]( https://www.kaggle.com/datasets/sohier/30-years-of-european-wind-generation). Hydro was acquired at: [Hydro](https://www.kaggle.com/datasets/mahbuburrahman2020/europe-green-electricity-generation-consumption).

### Data Preparation
We supply all processed datasets and put them under `./data`, the folder tree is shown below:
```
|-data
| |-ECL
| | |-ECL.csv
| |
| |-Hydro_BXX
| | |-Hydro_BXX.csv
| |
| |-Solar
| | |-solar_AL.csv
| |
| |-Wind
| | |-Wind.csv
| |
| ...
```

The processing details for the four datasets are as follows. We place ECL in the folder `./electricity` of [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing) (the folder tree in the link is shown as below) into folder `./data` and rename it from `./electricity` to `./ECL`. We rename the file of ECL from `electricity.csv` to `ECL.csv` and rename its last variable from `OT` to original `MT_321`. The processed file can be found at `./data/ECL/ECL.csv`
```
The folder tree in https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing:
|-autoformer
| |-electricity
| | |-electricity.csv
```
To standardize the data format, we convert the data file of [Solar](https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-) from 'solar_AL.txt' to 'solar_AL.csv'. We place the processed file into the folder `./data/Solar`. For convenience, we processed the Wind and Hydro datasets and you can obtain the processed files at `./data/Wind/Wind.csv` and `./data/Hydro_BXX/Hydro_BXX.csv`, respectively.

## Usage
Commands for training and testing HST of all datasets are in `./scripts/Main.sh`. 

More parameter information please refer to `main.py`.

We provide a complete command for training and testing SimTFV:

```
python -u main.py --data <data> --long_input_len <long_input_len>  --short_input_len <short_input_len> --pred_len <pred_len> --encoder_layers <encoder_layers> --decoder_layers <decoder_layers> --patch_size <patch_size> --d_model <d_model> --decoder_IN --learning_rate <learning_rate> --dropout <dropout> --batch_size <batch_size> --train_epochs <train_epochs> --itr <itr>  --train --patience <patience> --decay <decay>
```

Here we provide a more detailed and complete command description for training and testing the model:

| Parameter name |                                          Description of parameter                                          |
|:--------------:|:----------------------------------------------------------------------------------------------------------:|
|      data      |                                              The dataset name                                              |
|   root_path    |                                       The root path of the data file                                       |
|   data_path    |                                             The data file name                                             |
|  checkpoints   |                                       Location of model checkpoints                                        |
|   long_input_len    |                                           Input length                                            |
|    short_input_len    |                                         Input length                                         |
|    pred_len   |                                        Prediction length                                    |
|     enc_in     |                                                 Input variable number                                                |
|    dec_out     |                                                Output variable number                                             |
|    d_model     |                                             Hidden dims of model                                             |
|  encoder_layers |                                           The num of layers in each encoder stage                                          |
|   decoder_layers  |                                      The num of layers in each decoder stage                                   |
|   patch_size   |                                The initial patch size in patch-wise attention                            |
|  Not_use_CV  |                                          Whether not to adopt the cross-variable attention in TVA                                       |
|  decoder_IN |                                           Whether to use decoder_IN                                        |
|    dropout     |                                                  Dropout                                                   |
|    num_workers     |                                                  Data loader num workers                                                   |
|      itr       |                                             Experiments times                                              |
|  train_epochs  |                                      Train epochs of the second stage                                      |
|   batch_size   |                         The batch size of training input data                          |
|   decay   |                         Decay rate of learning rate per epoch                         |
|    patience    |                                          Early stopping patience                                           |
| learning_rate  |                                          Optimizer learning rate                                           |


## Results
The experiment parameters of each dataset are formated in the `Main.sh` files in the directory `./scripts/`. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better mse results or draw better prediction figures. We present the multivariate forecasting results of the four datasets in Figure 2.

<p align="center">
<img src="./img/results.jpg" height = "269" weight = "1036" alt="" align=center />
<br><br>
<b>Figure 2.</b> Multivariate forecasting results.
</p>



## Contact
If you have any questions, feel free to contact Li Shen through Email (shenli@buaa.edu.cn) or Github issues. Pull requests are highly welcomed!
