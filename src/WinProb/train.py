import os

from src.WinProb.state_dataset import StateDataset
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.Agents.neural_networks.hybrid_net_win_loss import HybridNet
from src.Agents.neural_networks.cnn_pytorch import BasicCNN
import wandb

def main():
    # WandB part is adapted from https://wandb.ai/site/articles/intro-to-pytorch-with-wandb

    # WandB – Initialize a new run
    WANDB_PROJECT_NAME: str = "rl-dl-lab"
    ENTITY_NAME_ENV_NAME: str = "WANDB_ENTITY"
    ENTITY_NAME = "hasham"
    wandb.init(entity=ENTITY_NAME, project=WANDB_PROJECT_NAME)

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config          # Initialize config
    config.batch_size = 128          # input batch size for training (default: 64)
    config.test_batch_size = 512    # input batch size for testing (default: 1000)
    config.epochs = 8             # number of epochs to train (default: 10)
    config.lr = 0.001               # learning rate (default: 0.01)
    config.no_cuda = False         # disables CUDA training
    config.log_interval = 10     # how many batches to wait before logging training status


    dim_dict = {'maps': 6,
                'scalars': 24}

    hybrid_net = HybridNet(dim_dict, BasicCNN)

    wandb.watch(hybrid_net, log="all")

    train_data = DataLoader(StateDataset('../state_tensors/states'), batch_size=config.batch_size, shuffle=True, num_workers=5)
    test_data = DataLoader(StateDataset('../state_tensors/test'), batch_size=config.test_batch_size, shuffle=False, num_workers=5)

    train_count = len(StateDataset('../state_tensors/states'))
    test_count = len(StateDataset('../state_tensors/test'))

    model_save_path = os.path.abspath('../state_tensors/best_reward_NN.model')

    # Optmizer and loss function
    optimizer=Adam(hybrid_net.parameters(), lr=config.lr, weight_decay=0.0001)
    loss_function=nn.CrossEntropyLoss(weight=torch.tensor([.80, 1]))


    # Model training and saving best model

    best_accuracy = 0.0

    print("starting epochs...")

    for epoch in range(config.epochs):

        # Evaluation and training on training dataset
        hybrid_net.train()
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (maps, scalars, winners) in enumerate(train_data):
            if torch.cuda.is_available():
                maps = Variable(maps.cuda())
                scalars = Variable(scalars.cuda())
                winners = Variable(winners.cuda())

            optimizer.zero_grad()

            outputs = hybrid_net(maps, scalars)
            loss = loss_function(outputs, winners)

            wandb.log({'train loss from loop': loss})

            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data * (maps.size(0) + scalars.size(0))
            _, prediction = torch.max(outputs.data, 1)

            train_accuracy += int(torch.sum(prediction == winners.data))

        train_accuracy = train_accuracy / train_count
        train_loss = train_loss / train_count

        # Evaluation on testing dataset
        hybrid_net.eval()

        test_accuracy = 0.0
        test_loss = 0.0
        for i, (maps, scalars, winners) in enumerate(test_data):
            if torch.cuda.is_available():
                maps = Variable(maps.cuda())
                scalars = Variable(scalars.cuda())
                winners = Variable(winners.cuda())

            outputs = hybrid_net(maps, scalars)
            _, prediction = torch.max(outputs.data, 1)
            test_accuracy += int(torch.sum(prediction == winners.data))

            loss = loss_function(outputs, winners)
            test_loss += loss.cpu().data * (maps.size(0) + scalars.size(0))

        test_accuracy = test_accuracy / test_count
        test_loss = test_loss / test_count

        wandb.log({
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Train Loss": train_loss,
            "Test Loss": test_loss})

        print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
            train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

        # Save the best model
        if test_accuracy > best_accuracy:
            torch.save(hybrid_net.state_dict(), model_save_path)
            wandb.save('best_reward_NN.h5')
            best_accuracy = test_accuracy

        # save model after each epoch
        torch.save(hybrid_net.state_dict(), os.path.join(os.path.abspath('../state_tensors/'),f'model_{epoch}.pth'))
        torch.save(hybrid_net.state_dict(), os.path.join(os.path.abspath('../state_tensors/'),f'model_{epoch}.model'))
        wandb.save('reward_NN.h5')


if __name__ == "__main__":
    main()