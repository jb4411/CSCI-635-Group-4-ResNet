import time
from enum import Enum
from typing import List, Any, Type, Union

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import BatchSampler
from torch.nn.modules.loss import _Loss

from torchvision import datasets, models, transforms
from torchvision.datasets import VisionDataset
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet

from labml import tracker, experiment, monit, logger
from labml_helpers.device import DeviceInfo

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
                  train_loader_shuffle=True, valid_loader_shuffle=False):
    if dataset == DataSet.CIFAR10:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          # Pad and crop
                                          transforms.RandomCrop(32, padding=4),
                                          # Random horizontal flip
                                          transforms.RandomHorizontalFlip(),
                                          #
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ]))
        val_data = datasets.CIFAR10('./data', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
    elif dataset == DataSet.STL10:
        train_data = datasets.STL10('./data', split="train", download=True,
                                    transform=transforms.Compose([
                                        # Pad and crop
                                        transforms.RandomCrop(96, padding=4),
                                        # Random horizontal flip
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
        val_data = datasets.STL10('./data', split="test", download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]))

    # Training and validation data loaders
    data_loaders = {
        Phase.TRAIN: torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=train_loader_shuffle),
        Phase.VALID: torch.utils.data.DataLoader(val_data, batch_size=valid_batch_size, shuffle=valid_loader_shuffle)
    }

    return data_loaders, (train_data, val_data, train_loader_shuffle, valid_loader_shuffle)


class Trainer:
    # Name of this run
    run_name: str
    # Model
    model: ResNet
    # Number of layers in the ResNet model
    num_layers: int
    # Type of block used in the ResNet model
    block_type: Type[Union[BasicBlock, Bottleneck, None]] = None
    # Dataset to use (sets both train_data and val_data)
    dataset: DataSet
    # Train dataset
    train_data: VisionDataset
    # Train batch size
    train_batch_size: int = 32
    # Whether train data is reshuffled every epoch
    train_loader_shuffle = True
    # Valid dataset
    val_data: VisionDataset
    # Valid batch size
    valid_batch_size: int = 128
    # Whether valid data is reshuffled every epoch
    valid_loader_shuffle = False
    # Loss function
    criterion: _Loss = nn.CrossEntropyLoss()
    # Optimizer
    optimizer: Optimizer
    # Optimizer Learning rate
    lr = 0.001
    # Optimizer momentum
    momentum = 0.9
    # Optimizer weight_decay
    weight_decay = 0.0001
    # interval at which training results should be logged
    train_log_interval: int = 10
    # device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Local webapi url or LabML token
    token = 'http://localhost:5005/api/v1/track?'
    # internal
    _device_info: DeviceInfo
    _data_loaders: dict
    _train_seen: int = 0
    _train_correct: int = 0
    _valid_seen: int = 0
    _valid_correct: int = 0
    _base_conf: dict
    _conf_override: dict

    def __init__(self, dataset: DataSet, num_layers: int, run_name=None):
        self._device_info = DeviceInfo(use_cuda=torch.cuda.is_available(), cuda_device=0)
        self.dataset = dataset
        self.num_layers = num_layers
        if run_name is None:
            self.run_name = f"ResNet{num_layers} - {str(self.dataset).replace('DataSet.', '')}"
        else:
            self.run_name = run_name
        self._data_loaders, cfg = setup_dataset(self.dataset, self.train_batch_size, self.valid_batch_size)
        self.train_data, self.val_data, self.train_loader_shuffle, self.valid_loader_shuffle = cfg
        self.model, self.layers, self.block_type = get_model(self.num_layers, self.device, block_type=self.block_type)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        tracker.set_scalar("loss.*", True)
        tracker.set_scalar("accuracy.*", True)

        tracker.set_scalar("train_diff", True)
        tracker.set_scalar("val_diff", True)

        self.create_conf()

    def create_conf(self):
        # TODO - WIP: need to correctly implement saving config data to the run info
        self._base_conf = {

        }
        self._conf = {
            "block_type": self.block_type,
            "block_type.bottlenecks": None if self.block_type == BasicBlock else [64, 128, 256, 512],
            "dataset_name": str(self.dataset).replace("DataSet.", ""),
            "device": self._device_info,
            # "device.device_info": ,
            # "first_kernel_size": ,
            # "inner_iterations": ,
            "loss_func": self.criterion,
            # "mode": ,
            "model": self.model,
            "n_blocks": self.layers,
            # "n_channels": ,
            "optimizer": self.optimizer,
            "optimizer.learning_rate": self.lr,
            "optimizer.momentum": self.momentum,
            "optimizer.weight_decay": self.weight_decay,
            "train_dataset": self.train_data,
            "train_batch_size": self.train_batch_size,
            "train_loader_shuffle": self.train_loader_shuffle,
            "valid_dataset": self.val_data,
            "valid_batch_size": self.valid_batch_size,
            "valid_loader_shuffle": self.valid_loader_shuffle,
        }



    def train_model(self, num_epochs: int = 10, adjust_lr=False):
        self._conf["epochs"] = num_epochs
        if num_epochs == 1:
            text = " (1 epoch)"
        else:
            text = f" ({num_epochs} epochs)"

        t_range = range(len(self._data_loaders[Phase.TRAIN]))
        v_range = range(len(self._data_loaders[Phase.VALID]))
        t_data = list(self._data_loaders[Phase.TRAIN])
        v_data = list(self._data_loaders[Phase.VALID])
        t_acc = [0.0, 0.0, 0.0]
        v_acc = [0.0, 0.0, 0.0]
        acc_idx = 0
        experiment.create(name=self.run_name + text)

        with experiment.record(name=self.run_name + text, token=self.token, exp_conf=self._conf):
            for epoch in monit.loop(range(num_epochs)):
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
                    self.step(inputs, labels, phase, idx)

                tracker.save()
                tracker.new_line()

                if adjust_lr:
                    t_acc[acc_idx] = self._train_correct / self._train_seen
                    v_acc[acc_idx] = self._valid_correct / self._valid_seen
                    acc_idx = (acc_idx + 1) % 3
                    tracker.add({"train_diff": abs(min(t_acc) - max(t_acc)), "val_diff": abs(min(v_acc) - max(v_acc))})
                    if (abs(min(t_acc) - max(t_acc)) <= 0.005) and (abs(min(t_acc) - max(t_acc)) <= 0.005):
                        adjust_lr = False
                        new_lr = self.lr / 10
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.log(f"Learning rate adjusted from {self.lr} to {new_lr}.")

    def step(self, inputs, labels, phase: Phase, batch_idx: int):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == Phase.TRAIN):
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            if phase == Phase.TRAIN:
                loss.backward()
                self.optimizer.step()

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
        layers (List[int]): A list that contains the number of blocks for each layer.
        num_classes (int, optional): The number of classes for the classification task. Defaults to 1000.
        block_type (Type[Union[BasicBlock, Bottleneck]]): The type of block to use
    Returns:
        ResNet: The ResNet model.
    """
    if block_type is None:
        # If the number of blocks is less than or equal to 2, use BasicBlock, else use Bottleneck
        block = BasicBlock if max(layers) <= 2 else Bottleneck
    else:
        block = block_type

    # Create the ResNet model
    model = _resnet(block, layers, weights=None, num_classes=num_classes, **kwargs)

    return model


def calculate_total_layers(layers: List[int], block: Type[Union[BasicBlock, Bottleneck]]) -> int:
    """Calculate the total number of layers in a ResNet model.

    Args:
        block (Type[Union[BasicBlock, Bottleneck]]): The block type, either BasicBlock or Bottleneck.
        layers (List[int]): A list that contains the number of blocks for each layer.

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


def get_model(num_layers, device, block_type: Type[Union[BasicBlock, Bottleneck]] = None,
              **kwargs: Any) -> (ResNet, List[int], Type[Union[BasicBlock, Bottleneck]]):
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


def main():
    # Number of epochs
    num_epochs = 10
    # Dataset
    dataset = DataSet.STL10
    # Number of layers for the resnet model
    num_layers = 101

    trainer = Trainer(dataset, num_layers)
    trainer.train_batch_size = 32
    trainer.valid_batch_size = 32
    trainer.lr = 0.0001

    start = time.perf_counter()
    trainer.train_model(num_epochs, adjust_lr=False)
    end = time.perf_counter()
    print(f"Training took {end - start}")


if __name__ == '__main__':
    main()
