# NMD_scorer: an Enformer inspired model to predict whether all copies of one mRNA will be degraded by nonsense-mediated decay (1), none will (0) or a degree in between.


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
