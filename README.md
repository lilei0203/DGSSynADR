# DGSSynADR：Predicting synergistic anticancer drug combination based on low-rank global attention mechanism and bilinear predictor

## Abstract

Drug combination therapy has exhibited remarkable therapeutic efficacy in various complex diseases and gradually become a promising strategy for clinical treatment. However, screening novel synergistic drug combinations is challenging due to the huge number of possible drug combinations. Recently, computational methods based on deep learning models have emerged as an efficient way to predict synergistic drug combinations. Here, we propose a new deep learning method based on the global structured features of drugs and proteins, DGSSynADR, to predict synergistic anti-cancer drug combinations. To better represent the chemical structures and characteristics of drugs, DGSSynADR constructs a heterogeneous graph by integrating the PPI network, the drug-target interaction network and multi-omics data, utilizes a low-rank global attention (LRGA) model to perform global weighted aggregation of graph nodes and learn the global structured features of drugs and proteins, and then feeds the embedded drug and protein features into a bilinear predictor to predict the synergy scores of drug combinations in different cancer cell lines. DGSSynADR trains the entire network end-to-end using the loss function Smooth L1. Specifically, LRGA network brings better model generalization ability, and effectively reduces the complexity of graph computation. The bilinear predictor facilitates the dimension transformation of the features and fuses the feature representation of the two drugs to improve the prediction performance of the model. The loss function Smooth L1 effectively avoids the gradient explosion, contributing to better model convergence. To validate the performance of DGSSynADR, we compare it with seven competitive methods, including two advanced deep learning based methods and four classical machine learning methods. The comparison results demonstrate that DGSPreSyn achieves better performance evaluated by all four evaluation metrics, including MSE, RMSE, $R^{2}$ and PCC. Meanwhile, the prediction of DGSSynADR is validated by previous studies through case studies. Furthermore, detailed ablation experiments demonstrate that the one-hot coding drug feature, the LRGA model and the bilinear predictor play a key role in improving the prediction performance.

![图示](https://user-images.githubusercontent.com/90454740/202649017-df81300e-a69d-4087-86f1-3ab4b15ca221.jpg)


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

## Download the Big Datasets and copy to the correct catalogue

Link: 

https://pan.baidu.com/s/1eRP_t__YBCMo_jntH6TRXA?pwd=3b2z 

https://pan.baidu.com/s/1QG29gksVoaZAIuGjD0cCaw?pwd=c77m 

https://pan.baidu.com/s/1nBCyN2IALkY1gKIJ8lG9nw?pwd=v4le 

https://pan.baidu.com/s/1issTmnVUr4m5QWVMU2MhDA?pwd=h26s 

## Running

python train_loss.py 

All configuaration are in XX_test.py




