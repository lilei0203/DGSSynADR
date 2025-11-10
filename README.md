# DGSSynADR：Predicting synergistic anticancer drug combination based on low-rank global attention mechanism and bilinear predictor

In the "Comparison between different types of models" experiment of the study "A Review of Deep Learning Approaches for Drug Synergy Prediction in Cancer", DGSSynADR is retrained using the DrugComb dataset, and the hyperparameters are readjusted to ensure a fair comparison under consistent experimental conditions:

* batcisize=1024   
* learning rate= 0.00001    
* epoch=270


## Install Application:

1.Anaconda (https://www.anaconda.com/)

2.Git(https://github.com/)

## Enviorment Setup:

1.Create a new conda enviorment:

conda env create -f DGSSynADR.yml

Or 

conda create --name DGSSynADR

2.Acitivate this enviorment:

conda activate DGSSynADR

3.Install the required packages

Conda install -r requirements.txt

The data preprocessing files related to this project are too large to include here and can be downloaded separately from Baidu Netdisk.
Link: https://pan.baidu.com/s/1eRP_t__YBCMo_jntH6TRXA?pwd=3b2z 
https://pan.baidu.com/s/1QG29gksVoaZAIuGjD0cCaw?pwd=c77m 
https://pan.baidu.com/s/1nBCyN2IALkY1gKIJ8lG9nw?pwd=v4le 
https://pan.baidu.com/s/1issTmnVUr4m5QWVMU2MhDA?pwd=h26s


## Running

python train_loss.py 

All configuaration are in XX_test.py




