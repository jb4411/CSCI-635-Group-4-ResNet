from flexible_resnet import DataSet, Trainer


def main():
    # Number of epochs
    num_epochs = 50
    # Dataset
    dataset = DataSet.STL10
    # Number of layers for the resnet model
    num_layers = 18

    # Create the trainer
    trainer = Trainer(dataset, num_layers)

    # Start training the model
    trainer.train_model(num_epochs)


if __name__ == '__main__':
    main()
