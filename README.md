# NMD_scorer
## A model to predict whether all copies of an mRNA sequence will be degraded by nonsense-mediated decay (1), none will (0) or a degree in between.

![alt text](https://github.com/guillametlucia/NMD_scorer/blob/main/fig_github.png)
Blocks surrounded by dashed square are inspired by the Enformer model (Avsec et al., 2021). 

Input: one-hot encoded (padded, if necessary) nucleotide sequence of length 20kbp and mask (mask out padding). 
eg [batch_size, channels = 4, sequence_length = 20000]
 
1) Convolutions and attention pooling
2) Transformer layers
3) NMD efficacy scoring head
   - Global pooling layer
   - Fully connected layer
   - Batch Normalisation
   - Sigmoid activation

Output: NMD efficacy score (NES) between 0 and 1

## NMDscorer.py: 
The model.

## get_genomic_features.r: 
Retrieve features from each mRNA sequence for interpretability.

## optuna_hyperparamtuning.py: 
File to tune model hyperparameters with Optuna.

## example_train_evaluate.ipynb: 
Example notebook of how to train and evaluate model.

## Bibliography
Avsec, Z., V. Agarwal, D. Visentin, J. Ledsam, A. Grabska-Barwinska, K. Taylor, Y. Assael, J. Jumper, P. Kohli, and D. Kelley (2021). Effective gene expression prediction from sequence by integrating long-range interactions. Nature Methods 18, 1196–1203.
