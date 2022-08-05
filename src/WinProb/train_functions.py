import torch
from torch.autograd import Variable

def train(model, loss_function, train_loader, optimizer):
    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (maps, scalars, winners) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(maps.cuda())
            scalars = Variable(scalars.cuda())
            winners = Variable(winners.cuda())

        optimizer.zero_grad()

        outputs = model(maps, scalars)
        loss = loss_function(outputs, winners)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * (maps.size(0) + scalars.size(0))
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == winners.data))