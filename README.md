# NMD_scorer

Input: one-hot encoded (padded, if necessary) nucleotide sequence of length 20kbp and mask. 

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
   - Sigmoid activation

Output: NMD efficacy score (NES) between 0 and 1
