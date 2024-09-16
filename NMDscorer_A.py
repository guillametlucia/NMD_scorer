# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import gc
import numpy as np
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable, List


class ConvBlock(nn.Module):
  """ Convolutional block"""


  def __init__(self, in_channels: int ,
               out_channels: int,
               kernel_size:int ):
    """
    Performs batch normalisation, GELU activation and convolution.

    Args:
    in_channels: number of channels in the input.
    out_channels: number of channels after performing convolution.
    kernel_size: size of kernel.
    """
    super().__init__()

    self.bn_gelu_conv = nn.Sequential(
        nn.SyncBatchNorm(in_channels),
        nn.GELU(),
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same')
    )

  def forward(self,x: torch.Tensor):
    return self.bn_gelu_conv(x)

class Residual(nn.Module):
  """ Residual block """
  def __init__(self, module: nn.Module):
    """
    Performs skip connection
    """
    super().__init__()
    self.module = module
  def forward(self, x: torch.Tensor, *args, **kwargs):
    output = self.module(x, *args, **kwargs)
    if isinstance(output, tuple):
       module_out, module_weights = output
       assert x.shape == module_out.shape, 'Cannot add tensors of different size'
       return x + module_out, module_weights
    else:
      assert x.shape == output.shape, 'Cannot add tensors of different size'
      return x + output



class AttentionPooling1D(nn.Module):
    """Attention pooling operation."""
    def __init__(self, pool_size: int =2,
                  w_init_scale: int =2):
        """
        Applies attention-based pooling to 1-D sequences.
        Weights are computed for each channel separately.

        Args:
        pool_size: Pooling size
        w_init_scale = Weight initialisation scale.
        """
        super().__init__()

        self.pool_size = pool_size
        self.w_init_scale = w_init_scale
        self.logit_linear = None
        self.initialized = False

    def initialize(self, num_channels: int):
        self.logit_linear = nn.Conv2d(num_channels, num_channels,1, bias=False)
        nn.init.dirac_(self.logit_linear.weight)

        with torch.no_grad():
          self.logit_linear.weight.mul_(self.w_init_scale)

        self.initialized = True
    def reshape(self, x: torch.Tensor):
        batch_size, num_channels, length = x.shape
        x = x.view(batch_size, num_channels, length // self.pool_size, self.pool_size)
        return x

    def forward(self, x: torch.Tensor):
        batch_size, num_channels, length = x.shape
        remainder = length % self.pool_size != 0
        needs_padding = remainder >0
        if needs_padding:
          print('padding')
          pad_length = self.pool_size - remainder
          x = F.pad(x, (0, pad_length), value=0)
          mask = torch.zeros((batch_size, 1, length), dtype = torch.bool, device= x.device)
          mask = F.pad(mask, (0, pad_length), value = True)

        if not self.initialized:
            self.initialize(num_channels)

        # Reshape input for pooling
        x = self.reshape(x)
        # Compute logits for softmax
        logits = self.logit_linear(x)

        if needs_padding:
          mask_value = -torch.finfo(logits.dtype).max
          logits = logits.masked_fill_(self.reshape(mask), mask_value)

        # Apply softmax to get weights
        weights = F.softmax(logits, dim=-1)
        del logits
        
        gc.collect()

        torch.cuda.empty_cache()
        # Weighted sum of the pooled features
        return (x * weights).sum(dim=-1)


def prepend_dims(x: torch.Tensor , num_dims: int) -> torch.Tensor:
    return x.view((1,) * num_dims + x.shape)

def positional_features_exponential(positions: torch.Tensor,
                                          feature_size: int,
                                          seq_length: Optional[int] = None,
                                          min_half_life: Optional[float] = 3.0) -> torch.Tensor:

    """Create exponentially decaying positional weights.

    Args:
      positions: Position tensor (arbitrary shape).
      feature_size: Number of basis functions to use.
      seq_length: Sequence length.
      min_half_life: Smallest exponential half life in the grid of half lives.

    Returns:
      A Tensor with shape [positions.shape, feature_size].
    """
    if seq_length is None:
        seq_length = torch.max(torch.abs(positions)) + 1
    # Grid of half lifes from [3, seq_length / 2] with feature_size distributed on the log scale.
    seq_length = torch.tensor(seq_length, dtype=torch.float)
    max_range = torch.log(torch.tensor(seq_length, dtype=torch.float)) / torch.log(torch.tensor(2.0))
    half_life = torch.pow(2.0, torch.linspace(min_half_life, max_range, feature_size, device = positions.device))
    half_life = prepend_dims((half_life), positions.dim())
    positions = torch.abs(positions)
    outputs = torch.exp(-torch.log(torch.tensor(2.0)) / half_life * positions.unsqueeze(-1))
    assert outputs.shape[-1] == feature_size, \
        f"Feature size mismatch: outputs' last dimension is {outputs.shape[-1]}, expected {feature_size}"
    assert outputs.shape[:-1] == positions.shape, \
       f"Shape mismatch: {outputs.shape} and {positions.shape}"
    return outputs

def positional_features_central_mask(positions: torch.Tensor,
                                     feature_size: int,
                                     seq_length: Optional[int] = None):
    """Central mask positional weights."""
    del seq_length  # Unused.
    center_widths = torch.pow(2.0, torch.arange(1, feature_size + 1, dtype=torch.float32))
    center_widths -=1
    center_widths = prepend_dims(center_widths, positions.dim())
    outputs = (center_widths > torch.abs(positions).unsqueeze(-1)).float()
    # Ensure the last dimension of outputs is feature_size
    assert outputs.shape[-1] == feature_size, \
        f"Feature size mismatch: outputs' last dimension is {outputs.shape[-1]}, expected {feature_size}"
    # Check compatibility for the rest of the dimensions
    assert outputs.shape[:-1] == positions.shape, \
        f"Shape mismatch: {outputs.shape} and {positions.shape}"
    return outputs

def gamma_pdf(x, concentration,rate):
    """ Gamma PDF p(x|concentration, rate)"""
    log_unnormalized_prob = torch.xlogy(concentration - 1, x) - rate * x
    log_normalization =  torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)




def positional_features_gamma(positions: torch.Tensor,
                              feature_size: int,
                              seq_length: Optional[int] = None,
                                stddev = None,
                                start_mean = None):
  """ Positional features computed using the gamma distribution."""
  if seq_length is None:
      seq_length = torch.max(torch.abs(positions)) + 1
  if stddev is None:
      stddev = seq_length/(2*feature_size)
  if start_mean is None:
      start_mean = seq_length/feature_size

  mean = torch.linspace(start_mean, seq_length, feature_size)
  mean = prepend_dims(mean, positions.dim())
  concentration = (mean/stddev)**2
  rate = mean/(stddev**2)
  probabilities = gamma_pdf((torch.abs(positions.unsqueeze(-1))).float(), concentration, rate)
  probabilities += 1e-8  # To ensure numerical stability.
  outputs = probabilities/ torch.max(probabilities, dim = 1, keepdim = True).values

  assert outputs.shape[-1] == feature_size, \
      f"Feature size mismatch: outputs' last dimension is {outputs.shape[-1]}, expected {feature_size}"
  # Check compatibility for the rest of the dimensions
  assert outputs.shape[:-1] == positions.shape, \
      f"Shape mismatch: {outputs.shape} and {positions.shape}"
  return outputs


def get_positional_feature_function(name):
  """Returns positional feature functions."""
  available = {
      'positional_features_exponential': positional_features_exponential,
      'positional_features_central_mask': positional_features_central_mask,
      'positional_features_gamma': positional_features_gamma
  }
  if name not in available:
    raise ValueError(f'Function {name} not available in {available.keys()}')
  return available[name]

def positional_features_all(positions: torch.Tensor,
                            feature_size: int,
                            seq_length: int):
  """Compute relative positional encodings.

  Each positional feature function will compute the same number of features, making up the total of feature_size.

  Args:
    positions: Tensor of relative positions of arbitrary shape.
    feature_size: Total number of basis functions.
    seq_length: Sequence length denoting the characteristic length that the individual positional features can use.
    """
  feature_functions = ['positional_features_exponential',
                        'positional_features_central_mask',
                        'positional_features_gamma']
  num_components = len(feature_functions) *2
  feature_functions = [get_positional_feature_function(name) for name in feature_functions]
  num_basis_per_class = feature_size // num_components
  embeddings = torch.concat(
      [f(torch.abs(positions), num_basis_per_class, seq_length) for f in feature_functions],
      dim=-1)
  embeddings = torch.concat(
      [embeddings, torch.sign(positions).unsqueeze(-1)*embeddings],
      dim=-1)

  # Ensure the last dimension of embeddings is feature_size
  assert embeddings.shape[-1] == feature_size, \
      f"Feature size mismatch: embeddings' last dimension is {embeddings.shape[-1]}, expected {feature_size}"
  # Check compatibility for the rest of the dimensions
  assert embeddings.shape[:-1] == positions.shape, \
      f"Shape mismatch: {embeddings.shape} and {positions.shape}"
  return embeddings

def relative_shift(x: torch.Tensor):
  """Shift the relative logits"""
  to_pad= torch.zeros_like(x[...,:1])
  x = torch.cat([to_pad, x], dim=-1)
  _ , num_heads, t1,t2 = x.shape
  x = x.view(-1, num_heads, t2, t1)
  x = x[:, :, 1:, :]
  x = x.view(-1, num_heads, t1, t2-1)
  x = x[:,:,:,:(t2+1)//2]
  return x


class MultiheadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, value_size: int,
                 key_size: int,
                 num_heads: int,
                 channels:int,
                 attention_dropout_rate: float,
                 positional_dropout_rate: float):

      """ Creates a multi-head attention module using TransformerXL relative attention style.

            Args:
              value_size: The size of each value embedding per head.
              key_size: The size of each key and query embedding per head.
              num_heads: The number of independent queries per timestep.
              attention_dropout_rate: Dropout rate for attention logits.
              positional_dropout_rate: Dropout rate for the positional encodings if
                relative positions are used.
    """
      super().__init__()
      self.value_size = value_size
      self.key_size = key_size
      self.num_heads = num_heads
      self.attention_dropout_rate = attention_dropout_rate
      self.positional_dropout_rate = positional_dropout_rate

      # Number of relative position features
      self.num_relative_position_features = channels // num_heads

      # Projection layers
      self.embedding_size = self.value_size * self.num_heads #i.e. channels

      self.q_layer = nn.Linear(channels, self.key_size * self.num_heads, bias=False)

      self.k_layer = nn.Linear(channels, self.key_size * self.num_heads, bias=False)

      self.v_layer = nn.Linear(channels, self.embedding_size, bias=False)


      # Initialize weights
      nn.init.xavier_normal_(self.q_layer.weight)
      nn.init.xavier_normal_(self.k_layer.weight)
      nn.init.xavier_normal_(self.v_layer.weight)

      self.embedding_layer = nn.Linear(self.embedding_size, self.embedding_size)

      # Zero-initialize the final layer
      nn.init.zeros_(self.embedding_layer.weight)

      # Additional layer for relative positions
      self.r_k_layer = nn.Linear(self.num_relative_position_features, self.key_size * self.num_heads, bias=False)
      nn.init.xavier_normal_(self.r_k_layer.weight)

      self.r_w_bias = nn.Parameter(torch.empty(1, self.num_heads, 1, self.key_size))
      nn.init.xavier_normal_(self.r_w_bias.data)

      self.r_r_bias = nn.Parameter(torch.empty(1, self.num_heads, 1, self.key_size))
      nn.init.xavier_normal_(self.r_r_bias.data)

    def multihead_output(self, linear, x: torch.Tensor):
         output = linear(x)  # [B, T, H * K or V]
         batch_size, seq_len, _ = output.size()
         num_kv_channels = output.size(-1) // self.num_heads
         output = output.view(batch_size, seq_len, self.num_heads, num_kv_channels)
         return output.permute(0, 2, 1, 3)  # [B, H, T, K or V] split into different heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Initialise the projection layers
        # X currently [B, T, C] with C = channels, T = seq_len.
        seq_len, device = x.size(1), x.device

        # Compute the queries, keys and values as multi-head projections of the inputs
        q = self.multihead_output(self.q_layer, x)  # [B, H, T, Q]
        k = self.multihead_output(self.k_layer, x)  # [B, H, T, K]
        v = self.multihead_output(self.v_layer, x)  # [B, H, T, V]
        del x

        # Scale the attention logits
        q = q * (self.key_size ** -0.5)

        # Compute the relative position encodings
        distances = torch.arange(-seq_len + 1, seq_len, device= device, dtype=torch.float32).unsqueeze(0)
        positional_encodings = positional_features_all(
            positions=distances, # [1, 2T - 1]
            feature_size=self.num_relative_position_features, # C // H = V
            seq_length=seq_len) # T
        del distances

        positional_encodings = F.dropout(positional_encodings, p=self.positional_dropout_rate)

        r_k = self.multihead_output(self.r_k_layer, positional_encodings)
        del positional_encodings, seq_len


        content_logits = torch.einsum('bnqd,bnkd->bnqk', q, k)
        relative_logits = torch.einsum('bnqd,bnkd->bnqk', q + self.r_w_bias, r_k)
        del q, k, r_k

        relative_logits = relative_shift(relative_logits)
        logits = content_logits.add_(relative_logits)
        del content_logits, relative_logits
        gc.collect()
        torch.cuda.empty_cache()

        # Apply mask
        mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, logits.size(1), logits.size(2), -1)  # [B, num_heads, T, T]
        logits = logits.masked_fill(mask ==0, -1e9)
        del mask

        weights = F.softmax(logits, dim=-1)
        del logits

        weights = F.dropout(weights, p=self.attention_dropout_rate)

        output = torch.einsum('bnqk,bnvd->bnqd', weights, v)
        del v
        gc.collect()
        torch.cuda.empty_cache()
        output_transpose = output.permute(0, 2, 1, 3)
        attended_inputs = output_transpose.contiguous().view(output_transpose.size(0),output_transpose.size(1),self.embedding_size)
        output = self.embedding_layer(attended_inputs)

        if self.training:
           return output
        # If evaluating, return average of weights across all heads for model interpretability.
        else:
          return output, weights.mean(dim=1)  # [B, T, T]


class MultiheadAttentionSequential(nn.Module):
    """Multi-head attention applied sequentially with layer normalization and dropout"""
    def __init__(self, value_size, key_size, num_heads, channels, attention_dropout_rate, positional_dropout_rate):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=channels)
        self.multihead_attention = MultiheadAttention(
            value_size,
            key_size,
            num_heads,
            channels,
            attention_dropout_rate=attention_dropout_rate,
            positional_dropout_rate=positional_dropout_rate
        )
        self.dropout = nn.Dropout(positional_dropout_rate)

    def forward(self, x, mask=None):
        x = self.layer_norm(x)
        if self.training:
           x = self.multihead_attention(x, mask=mask)
           x = self.dropout(x)
           return x
        # If evaluating, return average of weights across all heads for model interpretability. Don't apply dropout.
        else:
           x, weights = self.multihead_attention(x, mask=mask)
           return x, weights

class Transformer(nn.Module):
  """ Transformer block: MHA + feedforward layer"""
  def __init__(
    self,
    channels: int,
    value_size: int,
    key_size: int,
    num_heads: int,
    attention_dropout_rate: float,
    positional_dropout_rate: float,
    dropout_rate: float):
    super().__init__()

    self.multiheadattention = Residual(
    MultiheadAttentionSequential(
             value_size,
               key_size,
               num_heads,
               channels,
               attention_dropout_rate,
               positional_dropout_rate),
     )

    self.feedforward = Residual(nn.Sequential(
        nn.LayerNorm(normalized_shape = channels),
        nn.Linear(in_features = channels  ,out_features = channels*2),
        nn.Dropout(dropout_rate),
        nn.ReLU(),
        nn.Linear(in_features = channels*2  ,out_features = channels),
        nn.Dropout(dropout_rate)
     ) )


  def forward(self, x, mask):
    if self.training:
      x = self.multiheadattention(x, mask)
      x = self.feedforward(x)
      return x
    # If evaluating, return average of weights across all heads for model interpretability.
    else:
      x, weights = self.multiheadattention(x,mask)
      x = self.feedforward(x)
      return x, weights

def exponential_linspace_int(start:int , end: int, num: int, divisible_by: int):
  """Exponentially increasing values of integers. """
  def round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [round(start * base**i) for i in range(num)]

class NMDscorer(nn.Module):
  """ Main model."""
  def __init__(self,
               channels: 96,
               num_heads: int = 8,
               num_conv: int =4,
               window_size: int = 32,
               num_transformer: int= 1, 
               dropout_rate: float = 0.4,
               attention_dropout_rate: float = 0.05,
               positional_dropout_rate: float = 0.01,
               key_size: int = 32,
               relative_position_functions: List[str] = [
            'positional_features_exponential',
            'positional_features_central_mask',
            'positional_features_gamma'
        ]
               ):

    """ NMD scorer model.
    Args:
      channels: number of desired output channels.
      num_heads: number of attention heads.
      num_conv: number of times go through convolution tower.
      window_size:  number of bp contained in each position of sequence length after convolutions
      num_transformer: number of transformer layers.
      dropout_rate: dropout rate for fully connected layer in scoring head.
      attention_dropout_rate: dropout rate for attention logits.
      positional_dropout_rate: dropout rate for positional encodings.
      key_size: size of each key and query embedding per head.
      relative_position_functions: list of function names used for relative positional biases.
    """

    super().__init__()


    assert channels % num_heads ==0, (f'channels needs to be divisible by {num_heads}')
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Attention arguments
    value_size = channels//num_heads

    # Stem block: convolutional layer with skip connection + attention pooling.
    self.stem = nn.Sequential(
         nn.Conv1d(in_channels = 4, out_channels = channels//2 , kernel_size = 15,padding = 'same'),
         Residual(ConvBlock(in_channels = channels//2, out_channels = channels//2, kernel_size = 1)),
         AttentionPooling1D(pool_size =2))

    # Calculate number of output channels at each layer of the convolutional tower to end up with the desired 'channels' specified in the arguments.
    filter_list = [channels//2] + exponential_linspace_int(start = channels //2, end = channels, num = num_conv , divisible_by= window_size)


    # Convolutional tower: stacked convolutional layers with skip connection + attention pooling. Iterate num_conv times.
    self.conv_tower = nn.ModuleList([
        nn.Sequential(
            ConvBlock(in_channels = filter_list[i], out_channels = filter_list[i+1] , kernel_size = 5),
            Residual(ConvBlock(in_channels =filter_list[i+1] , out_channels = filter_list[i+1], kernel_size = 1 )),
            AttentionPooling1D(pool_size =2)
        )
        for i in range(len(filter_list)-1)
    ])

    # Transformer block
    self.transformer_block = nn.ModuleList([
       Transformer(channels = channels,
    value_size = value_size,
    key_size = key_size,
    num_heads = num_heads,
    attention_dropout_rate = attention_dropout_rate,
    positional_dropout_rate = positional_dropout_rate,
    dropout_rate = dropout_rate
)
        for _ in range(num_transformer)
    ])

    # NMD scoring head: global pooling + fully connected layer + sigmoid activation.
    self.scoring_head = nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(dropout_rate),
        nn.Linear(channels, 1),
        nn.Sigmoid(),
        nn.Flatten()
    )
    # self.global_pooling = nn.AdaptiveAvgPool1d(1)
    # self.dropout = nn.Dropout(dropout_rate)
    # self.fc1 = nn.Linear(channels, 1)
    # self.sigmoid = nn.Sigmoid()

    # Avg pooling layer
    self.pool = nn.AvgPool1d(kernel_size = 2, stride = 2)


  def forward(self, x: torch.Tensor, mask: torch.Tensor):
    # X is of shape [B, C, T]. Batch size, channels, sequence length
    # Mask [B, T]

    # Pass through convolutions
    x = self.stem(x)
    mask = self.pool(mask)
    for layer in self.conv_tower:
      x = layer(x)
      mask = self.pool(mask)

    x = x.permute(0,2,1) # change shape to [B,T,C] 

    # Pass mask through threshold. If value is greater than 0.5, then it is True, and attention will be applied to that position.
    mask = mask > 0.5
    
    all_attention_weights = []
    # Pass through transformer
    for layer in self.transformer_block:
      if self.training:
        x = layer(x, mask)
      else:
        # Returns sequence and attention weights for that layer if evaluating
        x, attention_weights = layer(x, mask)
        all_attention_weights.append(attention_weights)
    
    if not self.training:
      # Stack all attention weights into a single tensor
      all_attention_weights = torch.stack(all_attention_weights)
      # Compute average of all attention weights 
      avg_attention_weights = torch.mean(all_attention_weights, dim=0)


      # Get positions at which mask is True to interpret model
      # Initialize a tensor to store the first and last True indices for each batch
      indices = torch.full((mask.size(0), 2), -1, dtype=torch.long)
      del mask
      for i, batch in enumerate(mask):
          true_indices = torch.nonzero(batch).squeeze()
          if len(true_indices) > 0:
              indices[i, 0] = true_indices[0]
              indices[i, 1] = true_indices[-1]
          del true_indices
      
    
      # Returns predicted score, average attention across all layers, indices where non masked values are (first and last)
      return self.scoring_head(x.permute(0,2,1)).squeeze(-1), avg_attention_weights, indices
    
    
    
    return self.scoring_head(x.permute(0,2,1)).squeeze(-1)
