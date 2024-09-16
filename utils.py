import numpy as np
import json
import os
import dataclasses
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

from imitation.util import logger as imit_logger

class DictDataset(Dataset):
    def __init__(self, dataset: dict):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset['state'])
    def __getitem__(self, idx):
        output = {}
        for key in self.dataset.keys():
            output[key] = self.dataset[key][idx]
        return output

class RandomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, prefetch_factor=2, persistent_workers=False):
        super(RandomDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

    def __iter__(self):
        self.dataset.shuffle_data()
        return super(RandomDataLoader, self).__iter__()
    
    def sample(self):
        return next(iter(self))

class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

def prepare_dataset(dataset:dict, ratio:int=0.8):
    assert 'state' in dataset.keys(), 'states not in dataset'
    random_order = np.arange(len(dataset['state']))
    np.random.shuffle(random_order)
    train_set = {}
    valid_set = {}
    for key in dataset.keys():
        dataset[key] = np.array(dataset[key])[random_order]
        train_set[key] = dataset[key][:int(len(dataset['state'])*ratio)]
        valid_set[key] = dataset[key][int(len(dataset['state'])*ratio):]
    return train_set, valid_set

def read_config(file_path:str):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def log_to_file(sentence:str, filename:str):
    filename = 'logs/' + filename
    print(sentence)
    # create the file if it does not exist
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(sentence+'\n')
    else:
        with open (filename, 'a') as f:
            f.write(sentence+'\n')

def combine_datasets(dataset1:Union[dict, DictDataset], dataset2:dict):
    combined_dataset = {}
    if type(dataset1) == dict:
        for key in dataset1.keys():
            if key in dataset2.keys():
                # print(key)
                if type(dataset1[key]) == np.ndarray:
                    dataset1[key] = dataset1[key].tolist()
                dataset1[key].extend(dataset2[key])
                combined_dataset[key] = dataset1[key]
    elif type(dataset1) == DictDataset:
        for key in dataset1.dataset.keys():
            if key in dataset2.keys():
                # print(key)
                if type(dataset1.dataset[key]) == np.ndarray:
                    dataset1.dataset[key] = dataset1.dataset[key].tolist()
                dataset1.dataset[key].extend(dataset2[key])
                combined_dataset[key] = dataset1.dataset[key]
    combined_dataset = DictDataset(combined_dataset)
    return combined_dataset

@dataclasses.dataclass(frozen=True)
class ContInterventionMetrics:
    """Container for the different components of behavior cloning loss."""

    total_loss: torch.Tensor
    disc_loss: torch.Tensor
    cont_loss: torch.Tensor

@dataclasses.dataclass(frozen=True)
class DiscInterventionMetrics:
    """Container for the different components of behavior cloning loss."""

    total_loss: torch.Tensor


class Logger:
    def __init__(self, logger: imit_logger.HierarchicalLogger):
        """Create new logger.

        Args:
            logger: The logger to feed all the information to.
        """
        self._logger = logger
        self._tensorboard_step = 0
        self._current_epoch = 0

    def reset_tensorboard_steps(self):
        self._tensorboard_step = 0

    def log_epoch(
        self,
        epoch_number: int,
        epoch_loss: Union[ContInterventionMetrics, DiscInterventionMetrics],
    ):
        self._current_epoch = epoch_number
        self._logger.record("mile/epoch", epoch_number)
        
        # Log epoch loss
        for k, v in epoch_loss.__dict__.items():
            self._logger.record(f"mile/epoch/{k}", float(v) if v is not None else None)
        
        # # Dump epoch-level logs
        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

    def log_batch(
        self,
        batch_num: int,
        batch_size: int,
        training_metrics: Union[ContInterventionMetrics, DiscInterventionMetrics],
    ):
        self._logger.record("batch_size", batch_size)
        self._logger.record("mile/epoch", self._current_epoch)
        self._logger.record("mile/batch", batch_num)
        for k, v in training_metrics.__dict__.items():
            self._logger.record(f"mile/{k}", float(v) if v is not None else None)

        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

    def log_rollout(
        self,
        success_rate: float,
        init_success_rate: float,
    ):
        self._logger.record("mile/success_rate", success_rate)
        self._logger.record("mile/init_success_rate", init_success_rate)
        self._logger.record("mile/rollout_epoch", self._current_epoch)
        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

    def log_val(
        self,
        validation_metrics: Union[ContInterventionMetrics, DiscInterventionMetrics],
    ):
        for k, v in validation_metrics.__dict__.items():
            self._logger.record(f"mile/val/{k}", float(v) if v is not None else None)
        self._logger.record("mile/val_epoch", self._current_epoch)
        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

    def log_epoch_metrics(
        self,
        epoch_number: int,
        epoch_loss: Union[ContInterventionMetrics, DiscInterventionMetrics],
        validation_metrics: Union[ContInterventionMetrics, DiscInterventionMetrics] = None,
        success_rate: float = None,
        init_success_rate: float = None,
    ):
        self.log_epoch(epoch_number, epoch_loss)
        
        if validation_metrics is not None:
            self.log_val(validation_metrics)
        
        if success_rate is not None:
            self.log_rollout(success_rate, init_success_rate)

        # Dump epoch-level logs
        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_logger"]
        return state
