# NMD_scorer
## An Enformer inspired model to predict whether all copies of an mRNA sequence will be degraded by nonsense-mediated decay (1), none will (0) or a degree in between.

Input: one-hot encoded (padded, if necessary) nucleotide sequence of length 20kbp and mask (mask out padding). 

eg [batch_size, channels = 4, sequence_length = 20000]
[ ...
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [1. 0. 0. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]
 ...]
 
1) Convolutions and attention pooling
2) Transformer layers
3) NMD efficacy scoring head
   - Global pooling layer
   - Fully connected layer
   - Batch Normalisation
   - Sigmoid activation

Output: NMD efficacy score (NES) between 0 and 1

## NMDscorer.py: code for the model.

## getfeatures: for 
