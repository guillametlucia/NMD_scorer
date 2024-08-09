import logging
import os
import numpy as np
import pandas as pd
import optuna
import gc


#Joblib
from joblib import Parallel, delayed  # For parallel processing
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable, List
from torch.amp import autocast, GradScaler

from sklearn.model_selection import  train_test_split, RepeatedStratifiedKFold  # Cross-validation strategy to evaluate model performance

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import KBinsDiscretizer  # For discretizing continuous features, in this case NES

from torch.profiler import profile, record_function, ProfilerActivity

# import class from NMDscorer.py
from NMDscorer import NMDscorer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = 32

############################################################################################################
# Constants and Configuration
############################################################################################################

DATABASE_PATH = "/SAN/colcc/NMD_outputs/NMD_MSc_Project/"
DATA_FILE_TRAIN = 'training_NES.csv'
DATA_FILE_TEST = 'test_NES.csv'
STUDY_NAME = 'NMDscorer_optuna'
DATABASE_FILE = f'{DATABASE_PATH}{STUDY_NAME}.db'
STORAGE_URI = f'sqlite:///{DATABASE_FILE}'

import time

time_stamp = time.strftime("%Y%m%d-%H%M%S")
log_file_path = f'/SAN/colcc/NMD_outputs/NMD_MSc_Project/NMDscorer_optuna_{time_stamp}.log'

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)


# Configure logging
logging.basicConfig(level=logging.INFO,
                    filename=log_file_path,
                    filemode="w",
                    format="%(module)s - %(levelname)s - %(message)s")

###############################################################################
# Load data
###############################################################################

def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
  """One-hot encode sequence."""
  # Convert all chars in sequence to uppercase
  sequence = sequence.upper()
  def to_uint8(string):
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  return hash_table[to_uint8(sequence)].T

class PadToLength(nn.Module):
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, x: torch.Tensor):
        seq_len, target_len = x.shape[-1], self.target_length

        if seq_len < target_len:
            # Pad the sequence to the target length
            padding = target_len - seq_len
            x = F.pad(x, (0,padding), value=0)
            if padding % 128 > 63:
                padding +=1
            return x, torch.tensor(padding, dtype=torch.int64)
        else:
            return x, torch.tensor(0, dtype=torch.int64)

class NMDscorerDataset(Dataset):
    """
    PyTorch Dataset for NMD Scorer.
    """
    def __init__(self, dataframe, target_length=108928):
        """
        Initialize the dataset with the given data.

        dataframe : pd.DataFrame

        """
        self.target_length = PadToLength(target_length)
        dataframe.loc[:, 'EnsembleSequence'] = dataframe['EnsembleSequence'].apply(lambda x: torch.tensor(x, dtype=torch.int64))
        self.X = dataframe['EnsembleSequence'].values
        self.y = torch.tensor(dataframe['NES'].values, dtype=torch.float32)
    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Get the item at the given index.
        Parameters
        ----------
        idx : int
            Index of the item.

        Returns
        -------
        tuple
            Tuple containing the sequence, padding length, and target value.
        """
        sequence = self.X[idx]
        # Pad sequence if needed. Return the padded sequence and the padding length
        padded_sequence , pad_length = self.target_length(sequence)
        return padded_sequence , pad_length, self.y[idx]


def load_data(file_path):
    dataset = pd.read_csv(file_path)
    # One-hot encode the sequences
    dataset['EnsembleSequence'] = dataset['EnsembleSequence'].apply(one_hot_encode)
    return dataset


############################################################################################################
class NMDWrapper:
    """
    This class wraps the NMDscorer model and uses predefined hyperparameters for model training.
    Attributes
    ----------
    model : nmdscorer.
    """
    #Initiate class and parameters
    def __init__(self, channels, num_heads, num_conv, window_size,
                  num_transformer, dropout_rate, attention_dropout_rate,
                  positional_dropout_rate, key_size,
                  relative_position_functions):
        """
        Initialize the model with predefined hyperparameters.

        Parameters
        ----------
        channels: number of convolutional filters.
        num_heads: number of attention heads.
        num_conv
        window_size
        num_transformer: number of transformer layers.
        dropout_rate
        attention_dropout_rate
        positional_dropout_rate
        key_size
        relative_position_functions

        """
        self.params = {
        'channels': channels,
        'num_heads': num_heads,
        'num_conv': num_conv,
        'window_size': window_size,
        'num_transformer': num_transformer,
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': attention_dropout_rate,
        'positional_dropout_rate': positional_dropout_rate,
        'key_size': key_size,
        'relative_position_functions': relative_position_functions
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NMDscorer(**self.params).to(self.device)

    # Train model
    def train(self, train_dataloader, id, epochs, max_steps = 150000):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        train_dataloader : DataLoader
            DataLoader for the training data.
        """
        self.model.train()
        logging.info(f"Trial {id} started")

        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.MSELoss()

        def lr_lambda(step):
            if step < 5000:
                return step / 5000
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        patience = 5
        min_delta = 0.001
        best_loss = np.inf
        counter = 0
        step = 0
        early_stop = False

        scaler = GradScaler() if torch.cuda.is_available() else None

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            if step <= max_steps:
                for X_train, pad_length_train, y_train in train_dataloader:
                    with torch.profiler.profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        with_stack=True,
                        record_shapes=True,
                        profile_memory=True,
                        schedule=torch.profiler.schedule(wait=0, warmup=0, active=10, repeat=10)
                    ) as prof:
                        X_train, pad_length_train, y_train = X_train.to(self.device), pad_length_train.to(self.device), y_train.to(self.device)
                        X_train = X_train.float()
                        optimizer.zero_grad()
                        
                        if scaler is not None:
                            with autocast():
                                output = self.model(X_train, pad_length_train)
                                loss = criterion(output, y_train)
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                        else:
                            output = self.model(X_train, pad_length_train)
                            loss = criterion(output, y_train)
                            loss.backward()
                            optimizer.step()
                            
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.2)
                        
                        scheduler.step()
                        
                        step += 1

                        del X_train, pad_length_train, y_train, output
                        gc.collect()

                        prof.step()
                    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=40))

            best_loss = min(best_loss, loss.item())
            if epoch + 1 % 5 == 0:
                logging.info(f"Epoch {epoch + 1}/{epochs}, Step {step}, Loss: {loss.item()}")
                if loss.item() - best_loss > min_delta:
                    counter +=1
                    if counter >= patience:
                        early_stop = True
                else:
                    counter = 0
                early_stop = False
            if early_stop:
                logging.info(f"Early stopping at Epoch {epoch + 1}/{epochs}, Step {step}, Loss: {loss.item()}, Best loss was {best_loss} at epoch {epoch - patience + 1}")
                break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logging.info(f"Trial {id} finished with loss {best_loss} and parameters {self.params}")
        


    def predict_score(self, valid_dataloader, criterion = nn.MSELoss()):
        """
        Probability estimates for the val vector X_val.
        Parameters
        ----------
        valid_dataloader : DataLoader
            DataLoader for the validation data.
        criterion : loss function

        Returns
        -------
        predictions
            The predicted values. A list of tensors.

        float
            The mean squared error of the predictions.
        """
        self.model.eval()
        valid_loss = 0
        predictions = []
        with torch.no_grad():
            for X_valid, pad_length_valid, y_valid in valid_dataloader:
                X_valid, pad_length_valid, y_valid = X_valid.to(self.device), pad_length_valid.to(self.device), y_valid.to(self.device)
                X_valid = X_valid.float()
                predict = self.model(X_valid, pad_length_valid)
                predictions.append(predict)
                valid_loss += criterion(predict, y_valid).item()
                del X_valid, pad_length_valid, y_valid
        predictions = torch.cat(predictions, dim=0)
        return predictions, valid_loss/len(valid_dataloader)


############################################################################################################
# Run optimization and objective function for optuna
############################################################################################################

def objective_single_arg(train_idx, val_idx, fold_id, training_dataset, hyperparams, trial):
    """
    Objective function for a single set of hyperparameters in Optuna.

    Parameters
    ----------
    train_idx : array-like
        Indices for the training set.
    val_idx : array-like
        Indices for the val set.
    fold_id : int
        Fold identifier.
    training_dataset : pandas.DataFrame
    model : NMDWrapper
        The model to be trained and evaluated.

    Returns
    -------
    mse : float
        The MSE score for the predictions.
    """
    # Ensure each process has its own device allocation

    # Split the data into train and val sets

    print(f"Using device: {device} for trial: {trial.number}, fold: {fold_id}")

    # Convert into PyTorch DataLoader
    train = Subset(training_dataset, train_idx)
    val = Subset(training_dataset, val_idx)
    del training_dataset

    train_dataloader = DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=2, shuffle=False)

    trial_number = trial.number

    # Fit model
    model = NMDWrapper(**hyperparams)
    model.train(train_dataloader, id = trial_number, epochs=100)

    # Generate predictions
    mse, _ = model.predict_score(val_dataloader)

    # Report the score to Optuna

    return mse

def run_optimization(n_trials, training_dataset, cv_splits):
    """
    Run hyperparameter optimization using Optuna.

    Parameters
    ----------
    n_trials : int
        Number of trials for optimization.
    training_dataset : pandas.DataFrame
    cv : cross-validation generator
        Cross-validation generator.

    Returns
    -------
    optuna.study.Study
        The optimized study.
    """
    # Prepare arguments for parallel execution


    args = [dict(train_idx=train_idx, val_idx=val_idx, fold_id=fold_id, training_dataset=training_dataset)
            for fold_id, (train_idx, val_idx) in enumerate(cv_splits)]

    def objective(trial):#Objective function used by Optuna

        hyperparams = {
        'channels': trial.suggest_categorical('channels', [1152,1536, 2304]),
        'num_heads': 8,
        'num_conv': 6,
        'window_size': 128,
        'num_transformer': trial.suggest_categorical('num_transformer', [5,8,11]),
        'dropout_rate': trial.suggest_categorical('dropout_rate', [0.1,0.4,0.7]),
        'attention_dropout_rate': trial.suggest_categorical('attention_dropout_rate', [0.02, 0.05, 0.08]),
        'positional_dropout_rate': trial.suggest_categorical('positional_dropout_rate', [0.01, 0.03, 0.05]),
        'key_size': trial.suggest_categorical('key_size', [32, 64, 128]),
        'relative_position_functions': [   'positional_features_exponential',
            'positional_features_central_mask',
            'positional_features_gamma']
        }

  #      model = NMDWrapper(**hyperparams)# Create nmdscorer model

        # Use Joblib's Parallel to parallelize the execution
        scores = Parallel(n_jobs=-1)(delayed(objective_single_arg)(hyperparams = hyperparams,trial = trial, **arg) for arg in args)

        # Calculate the mean score and return it to optuna
        return np.mean(scores)


    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE_URI, direction='minimize',
                                    load_if_exists=True,
                                    sampler=sampler)

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    return study


############################################################################################################
# Main Function
############################################################################################################

def main():
    # Load training data.

    all_training_dataset = load_data(f'{DATABASE_PATH}{DATA_FILE_TRAIN}')

    # Discretize NES into quantiles and stratify the data in terms of tumour type and NES quantile
    kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    all_training_dataset['NES_q'] = kbins.fit_transform(all_training_dataset['NES'].values.reshape(-1, 1)).astype(int)

    bin_edges = kbins.bin_edges_[0]
    logging.info(f"Bin edges: {bin_edges}")

    # Split training into training and internal validation (heldout)
    training_dataset, internal_validation = train_test_split(
    all_training_dataset,
    test_size=0.3,
    random_state=42,
    stratify=all_training_dataset['NES_q'])

    del all_training_dataset

    # Save internal validation into csv and delete from here
    internal_validation.to_csv(f'{DATABASE_PATH}internal_validation_afteriv.csv', index=False)
    training_dataset.to_csv(f'{DATABASE_PATH}training_dataset_afteriv.csv', index=False)
    del internal_validation

    # Training dataset
    # From training dataset, drop unnecessary columns ie just keep the sequence and NES
    # Define the cross-validation scheme
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv_splits = list(cv.split(training_dataset, training_dataset['NES_q']))
    training_dataset = training_dataset[['EnsembleSequence', 'NES']]
    train_torch_dataset = NMDscorerDataset(training_dataset, target_length=108928
                                        )
    del training_dataset

    # Run optimization
    study = run_optimization(n_trials=400, training_dataset = train_torch_dataset, cv_splits = cv_splits)

    # Get the best hyperparameters

    best_hyperparams = {
                        'channels': study.best_params["channels"],
                        'num_heads': 8,
                        'num_conv': 5,
                        'window_size': 128,
                        'num_transformer': study.best_params["num_transformer"],
                        'dropout_rate': study.best_params["dropout_rate"],
                        'attention_dropout_rate': study.best_params["attention_dropout_rate"],
                        'positional_dropout_rate': study.best_params["positional_dropout_rate"],
                        'key_size':  study.best_params["key_size"],
                        'relative_position_functions': [   'positional_features_exponential',
            'positional_features_central_mask',
            'positional_features_gamma']
       }

    # Train the model with the best hyperparameters and save

    model_nmd_final = NMDWrapper(**best_hyperparams)

    train_dataloader = DataLoader(NMDscorerDataset(training_dataset), batch_size=batch_size, shuffle=True)

    model_nmd_final.train(train_dataloader, epochs=100)

    # Save model
    torch.save(model_nmd_final.state_dict(), f'{DATABASE_PATH}/Best_nmdscorer_seq.pt')


if __name__ == "__main__":
    main()
