"""
file: flexible_resnet.py
description: Methods and classes used to facilitate training various ResNet models.
language: Python 3.11
author: Jesse Burdick-Pless jb4411@rit.edu
author: Archit Joshi aj6082@rit.edu
author: Mona Udasi mu9326@rit.edu
author: Parijat Kawale pk7145@rit.edu
"""

import time
from enum import Enum
from typing import List, Any, Type, Union, Tuple

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import BatchSampler
from torch.nn.modules.loss import _Loss
from torchvision import datasets, models, transforms
from torchvision.datasets import VisionDataset
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet
from ignite.handlers.param_scheduler import PiecewiseLinear

from labml import tracker, experiment, monit
from labml_helpers.device import DeviceInfo

# Set up tracked metrics
tracker.set_scalar("loss.*", True)
tracker.set_scalar("accuracy.*", True)
tracker.set_scalar("learning_rate", True)


class DataSet(Enum):
    CIFAR10 = 1
    STL10 = 2


class StepType(Enum):
    PERF_STEP = 1
    APPROX_STEP = 2
    APPROX_DATA_STEP = 3
    PERF_DATA_STEP = 4


class Phase(Enum):
    TRAIN = 1
    VALID = 2


def setup_dataset(dataset: DataSet, train_batch_size, valid_batch_size,
                  train_loader_shuffle=True, valid_loader_shuffle=False, num_workers=0):
    """Preprocess the target dataset.

    Args:
        dataset: dataset to preprocess.
        train_batch_size: batch size for training.
        valid_batch_size: batch size for validation.
        train_loader_shuffle: whether training data is reshuffled every epoch, (default: True).
        valid_loader_shuffle: whether validation data is reshuffled every epoch, (default: False).
        num_workers: how many subprocesses to use for data loading, (default: 0).
               If 0, data will be loaded in the main process.

    Returns: data_loaders, (train_data, val_data, train_loader_shuffle, valid_loader_shuffle)
    """
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    if dataset == DataSet.CIFAR10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          # Pad and crop
                                          # transforms.RandomCrop(32, padding=4),
                                          # Random horizontal flip
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)
                                      ]))
        val_data = datasets.CIFAR10('./data', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                    ]))
    elif dataset == DataSet.STL10:
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2603, 0.2566, 0.2713)
        train_data = datasets.STL10('./data', split="train", download=True,
                                    transform=transforms.Compose([
                                        # Pad and crop
                                        # transforms.RandomCrop(96, padding=4),
                                        # Random horizontal flip
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                    ]))
        val_data = datasets.STL10('./data', split="test", download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                  ]))

    # Training and validation data loaders
    data_loaders = {
        Phase.TRAIN: torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=train_loader_shuffle,
                                                 num_workers=num_workers),
        Phase.VALID: torch.utils.data.DataLoader(val_data, batch_size=valid_batch_size, shuffle=valid_loader_shuffle,
                                                 num_workers=num_workers)
    }

    return data_loaders, (train_data, val_data, train_loader_shuffle, valid_loader_shuffle)


class Trainer:
    """A class to handle training ResNet models.

       Args:
           dataset: dataset to train on.
           num_layers: number of layers for the ResNet model.
           run_name: name of the run, (default=None). If None, run_name is set to "ResNet-{num_layers} - {dataset}".
           train_batch_size: number of training samples to load per batch, (default=32).
           valid_batch_size: number of validation samples to load per batch, (default=128).
           num_workers: how many subprocesses to use for data loading, (default: 0).
               If 0, data will be loaded in the main process.
           optimizer: the optimizer to use while training, (default: :class:`Adam<torch.optim.Adam>`).
           optimizer_momentum: optimizer momentum, (default: 0.9).
           optimizer_weight_decay: optimizer weight decay, (default: 0.0001).
           optimizer_lr: optimizer learning rate, (default: 0.001).
           lr_milestones: list of tuples (epoch, lr value), (default=None).
           custom_model: a custom ResNet model to train, (default=None).
       """
    # Name of this run
    run_name: str
    # Model
    model: ResNet = None
    # Number of layers in the ResNet model
    num_layers: int
    # Type of block used in the ResNet model
    block_type: Type[Union[BasicBlock, Bottleneck, None]] = None
    # Dataset to use (sets both train_data and val_data)
    dataset: DataSet = None
    # Train dataset
    train_dataset: VisionDataset
    # Train batch size
    train_batch_size: int = 32
    # Whether train data is reshuffled every epoch
    train_loader_shuffle = True
    # Valid dataset
    valid_dataset: VisionDataset
    # Valid batch size
    valid_batch_size: int = 128
    # Whether valid data is reshuffled every epoch
    valid_loader_shuffle = False
    # Loss function
    loss_func: _Loss = nn.CrossEntropyLoss()
    # Optimizer
    optimizer: Optimizer
    # Optimizer learning rate
    lr = 0.001
    # Milestones for optimizer learning rate schedule
    lr_milestones = None
    # Optimizer learning rate schedule
    lr_schedule = None
    # Optimizer momentum
    momentum = 0.9
    # Optimizer weight_decay
    weight_decay = 0.0001
    # Interval at which training results should be logged
    train_log_interval: int = 10
    # Device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Internal
    _device_info: DeviceInfo
    _data_loaders: dict
    _train_seen: int = 0
    _train_correct: int = 0
    _valid_seen: int = 0
    _valid_correct: int = 0
    _base_conf: dict
    _conf_override: dict

    def __init__(self, dataset: DataSet, num_layers: int, run_name=None, train_batch_size: int = 32,
                 valid_batch_size: int = 128, num_workers=0, optimizer: Type[Union[optim.SGD, optim.Adam]] = optim.Adam,
                 optimizer_momentum: float = 0.9, optimizer_weight_decay: float = 0.0001, optimizer_lr: float = 0.001,
                 lr_milestones: List[Tuple[int, float]] = None, custom_model=None):
        self._device_info = DeviceInfo(use_cuda=torch.cuda.is_available(), cuda_device=0)
        self.dataset = dataset
        self._data_loaders, cfg = setup_dataset(self.dataset, train_batch_size, valid_batch_size,
                                                num_workers=num_workers)
        self.train_dataset, self.valid_dataset, self.train_loader_shuffle, self.valid_loader_shuffle = cfg

        self.num_layers = num_layers
        if run_name is None:
            self.run_name = f"ResNet-{num_layers} - {str(self.dataset).replace('DataSet.', '')}"
        else:
            self.run_name = run_name

        if custom_model is None:
            self.model, self.layer_blocks, self.block_type = get_model(self.num_layers, self.device,
                                                                       block_type=self.block_type)
        else:
            self.model, self.layer_blocks, self.block_type = custom_model(self.device)

        if optimizer == optim.SGD:
            self.optimizer = optim.SGD(self.model.parameters(), lr=optimizer_lr, momentum=optimizer_momentum,
                                       weight_decay=optimizer_weight_decay)
        elif optimizer == optim.Adam:
            self.optimizer = optim.Adam(self.model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)

        if lr_milestones is not None:
            self.lr_milestones = lr_milestones
            self.lr_schedule = PiecewiseLinear(self.optimizer, "lr", milestones_values=lr_milestones)

        self._base_conf = self._create_conf()
        self._conf_override = dict()

        self.momentum = optimizer_momentum
        self.weight_decay = optimizer_weight_decay
        self.lr = optimizer_lr
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

    def _create_conf(self):
        conf = {
            "num_layers": self.num_layers,
            "block_type": self.block_type,
            "block_type.bottlenecks": None if self.block_type == BasicBlock else [64, 128, 256, 512],
            "dataset_name": str(self.dataset).replace("DataSet.", ""),
            "device": self._device_info,
            # "device.device_info": ,
            # "first_kernel_size": ,
            # "inner_iterations": ,
            "loss_func": self.loss_func,
            # "mode": ,
            "model": self.model,
            "layer_blocks": self.layer_blocks,
            # "n_channels": ,
            "optimizer": self.optimizer,
            "optimizer.learning_rate": self.lr,
            "optimizer.adjust_lr": False,
            "optimizer.lr_milestones": self.lr_milestones,
            "optimizer.lr_schedule": self.lr_schedule,
            "optimizer.momentum": self.momentum,
            "optimizer.weight_decay": self.weight_decay,
            "train_dataset": self.train_dataset,
            "train_batch_size": self.train_batch_size,
            "train_loader_shuffle": self.train_loader_shuffle,
            "valid_dataset": self.valid_dataset,
            "valid_batch_size": self.valid_batch_size,
            "valid_loader_shuffle": self.valid_loader_shuffle,
        }

        return conf

    def _conf_overrides(self):
        temp = self._create_conf()
        for param in temp:
            if self._base_conf[param] != temp[param]:
                self._conf_override[param] = temp[param]

    def train_model(self, num_epochs: int = 10, show_lr=False, adjust_lr=False):
        """Train the ResNet model.

        Args:
            num_epochs: number of epochs to train for.
            show_lr: if optimizer learning rate should be logged and graphed, (default: False).
            adjust_lr: if optimizer learning rate should be adjusted using learning rate milestones, (default: False).

        """
        start = time.perf_counter()

        if adjust_lr:
            self._base_conf["lr"] = None
            self._conf_override["optimizer.adjust_lr"] = True
            self._conf_override["optimizer.lr_milestones"] = self.lr_milestones
            self._conf_override["optimizer.lr_schedule"] = self.lr_schedule
            self.lr_schedule(0)
        else:
            self._base_conf["optimizer.lr_milestones"] = None
            self._base_conf["optimizer.lr_schedule"] = None

        self._conf_override["epochs"] = num_epochs
        self._conf_overrides()

        if num_epochs == 1:
            text = " (1 epoch)"
        else:
            text = f" ({num_epochs} epochs)"

        t_range = range(len(self._data_loaders[Phase.TRAIN]))
        v_range = range(len(self._data_loaders[Phase.VALID]))
        t_data = list(self._data_loaders[Phase.TRAIN])
        v_data = list(self._data_loaders[Phase.VALID])
        tracker.add({'loss.train': 0, 'accuracy.train': 0})
        tracker.add({'loss.valid': 0, 'accuracy.valid': 0})
        experiment.create(name=self.run_name + text)
        experiment.configs(self._base_conf, self._conf_override)
        with experiment.start():
            for epoch in monit.loop(range(num_epochs)):
                if adjust_lr:
                    self.lr_schedule(epoch + 1)
                if show_lr:
                    for param_group in self.optimizer.param_groups:
                        tracker.add({"lr": param_group['lr']})

                self._train_seen = 0
                self._train_correct = 0
                self._valid_seen = 0
                self._valid_correct = 0
                for p, idx in monit.mix(('Train', t_range), ('Valid', v_range)):
                    if p == 'Train':
                        phase = Phase.TRAIN
                        (inputs, labels) = t_data[idx]
                    else:
                        phase = Phase.VALID
                        (inputs, labels) = v_data[idx]
                    self._step(inputs, labels, phase, idx)

                tracker.save()
                tracker.new_line()

            if show_lr:
                for param_group in self.optimizer.param_groups:
                    tracker.add({"lr": param_group['lr']})

            tracker.save()

        end = time.perf_counter()
        show_training_time(start, end)

    def _step(self, inputs, labels, phase: Phase, batch_idx: int):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        with torch.set_grad_enabled(phase == Phase.TRAIN):
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.loss_func(outputs, labels)

            if phase == Phase.TRAIN:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        if phase == Phase.TRAIN:
            self._train_correct += torch.sum(preds == labels.data)
            self._train_seen += len(labels.data)
            # Increment the global step
            tracker.add_global_step(len(labels.data))
            # Store stats in the tracker
            tracker.add({'loss.train': loss, 'accuracy.train': self._train_correct / self._train_seen})
            if batch_idx % self.train_log_interval == 0:
                tracker.save()
        else:
            self._valid_correct += torch.sum(preds == labels.data)
            self._valid_seen += len(labels.data)
            tracker.add({'loss.valid': loss, 'accuracy.valid': self._valid_correct / self._valid_seen})
            tracker.save()


def generate_resnet(layers: List[int], num_classes: int = 10, block_type: Type[Union[BasicBlock, Bottleneck]] = None,
                    **kwargs: Any) -> ResNet:
    """Generate a ResNet model with an arbitrary number of layers.

    Args:
        layers: list that contains the number of blocks for each layer.
        num_classes: number of classes for the classification task, (default: 1000).
        block_type: The type of block to use.
    Returns:
        ResNet: The ResNet model.
    """
    if block_type is None:
        # If the number of blocks is less than or equal to 2, use BasicBlock, else use Bottleneck
        block = BasicBlock if max(layers) <= 2 else Bottleneck
    else:
        block = block_type

    # Create the ResNet model
    model = _resnet(block, layers, weights=None, num_classes=num_classes, progress=True, **kwargs)

    return model


def calculate_total_layers(layers: List[int], block: Type[Union[BasicBlock, Bottleneck]]) -> int:
    """Calculate the total number of layers in a ResNet model.

    Args:
        layers: list containing the number of blocks for each layer.
        block: block type, either BasicBlock or Bottleneck.

    Returns:
        int: The total number of layers in the ResNet model.
    """
    # Count of layers in a block
    if block == BasicBlock:
        block_layers = 2
    else:
        block_layers = 3

    # Count the total number of layers in blocks
    total_layers = sum(block_layers * x for x in layers)

    # Add 1 for the initial convolutional layer, 1 for the max pooling layer, and 1 for the final fully connected layer
    total_layers += 3

    return total_layers


def get_model(num_layers: int, device: torch._C.device, block_type: Type[Union[BasicBlock, Bottleneck]] = None,
              **kwargs: Any) -> (ResNet, List[int], Type[Union[BasicBlock, Bottleneck]]):
    """Get a ResNet model with {num_layers} layers.

    Args:
        num_layers: number of layers the returned ResNet model should have.
        device: device the model will be trained on.
        block_type: type of residual block the model should use, (default: None). If None, block_type is set to
            :class:`BasicBlock<torchvision.models.resnet.BasicBlock>` if num_layers is less than 50, if num_layers is
            greater than or equal to 50, block_type is set to :class:`Bottleneck<torchvision.models.resnet.Bottleneck>`.

    Returns: model, layers, block_type
    """
    if num_layers in [18, 34, 50, 101, 152]:
        pre_defined = {
            18: (models.resnet18, [2, 2, 2, 2], BasicBlock),
            34: (models.resnet34, [3, 4, 6, 3], BasicBlock),
            50: (models.resnet50, [3, 4, 6, 3], Bottleneck),
            101: (models.resnet101, [3, 4, 23, 3], Bottleneck),
            152: (models.resnet152, [3, 8, 36, 3], Bottleneck)
        }
        model, layers, block_type = pre_defined[num_layers]
        model = model(weights=None, **kwargs)
    else:
        if block_type is None:
            if num_layers < 50:
                block_type = BasicBlock
                block_layers = 2
            else:
                block_type = Bottleneck
                block_layers = 3
        else:
            if block_type == BasicBlock:
                block_layers = 2
            else:
                block_layers = 3
        target = num_layers - 3
        offset = target % block_layers
        target += block_layers - offset
        target -= block_layers * 2
        """
         50 -> 4, 6  ---  29 -> 12, 18
        101 -> 4, 23 ---  80 -> 12, 69
        152 -> 8, 36 --- 131 -> 24, 108
        """
        mid = target // 2
        layers = [block_layers, mid, target - mid, block_layers]
        actual = calculate_total_layers(layers, block_type)
        if num_layers != actual:
            print(f"Target = {num_layers} layers, actual = {actual} layers.")
        model = generate_resnet(layers, block_type=block_type, **kwargs)

    model = model.to(device)
    return model, layers, block_type


def show_training_time(start: float, end: float):
    """Display the amount of time training took in a human readable format.
    :class:`Adam<torch.optim.Adam>`
    Args:
        start: result returned by :class:`time.perf_counter()<time.perf_counter()>` before calling
            :class:`Trainer.train_model()<Trainer.train_model()>`.
        end: result returned by :class:`time.perf_counter()<time.perf_counter()>` after calling
            :class:`Trainer.train_model()<Trainer.train_model()>`.
    """
    text = "Training took"
    text = "Training time ="
    diff = end - start

    # milliseconds
    if diff < 1:
        diff *= 1000
        print(f"{text} {diff}ms")
        return

    mn = 60
    # seconds
    if diff < mn:
        print(f"{text} {diff}secs")
        return

    hr = 60 * mn
    # minutes and seconds
    if diff < hr:
        minutes = int(diff / mn)
        print(f"{text} {minutes} minute{'s' if minutes > 1 else ''} {diff % mn} seconds")
        return

    # hours, minutes and seconds
    hours = int(diff / hr)
    h_suffix = 's' if hours > 1 else ''
    minutes = int((diff % hr) / mn)
    m_suffix = 's' if minutes > 1 else ''
    print(f"{text} {hours} hour{h_suffix} {minutes} minute{m_suffix} {diff % mn} seconds")
