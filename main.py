"""
file: main.py
description: The main interface used to experiment with training various ResNet models.
    Created by Group 4 for CSCI 635 (Intro to Machine Learning) during the Summer 2023 semester.
language: Python 3.11
author: Jesse Burdick-Pless jb4411@rit.edu
author: Archit Joshi aj6082@rit.edu
author: Mona Udasi mu9326@rit.edu
author: Parijat Kawale pk7145@rit.edu
"""

from flexible_resnet import DataSet, Trainer


def main():
    # Number of epochs to train for
    num_epochs = 50
    # Dataset to train on
    dataset = DataSet.CIFAR10
    # Number of layers for the ResNet model
    num_layers = 18

    # Create the trainer
    trainer = Trainer(dataset, num_layers)

    # Start training the model
    trainer.train_model(num_epochs)


if __name__ == '__main__':
    main()
