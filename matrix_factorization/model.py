import torch
from tqdm import tqdm


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors, device):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True, device=device)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True, device=device)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(-1)


def train(model, device, train_loader, epoch, optimizer, criterion):
    model.train()

    epoch_loss = 0

    for batch_idx, (user, item, rating) in tqdm(enumerate(train_loader)):
        # Set gradients to zero
        optimizer.zero_grad()

        # Turn data into tensors
        rating = rating.to(dtype=torch.float, device=device)
        user = user.to(dtype=torch.long, device=device)
        item = item.to(dtype=torch.long, device=device)

        # Predict and calculate loss
        prediction = model(user, item)
        loss = criterion(prediction, rating)

        # Backpropagate
        loss.backward()

        # Update the parameters
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= int(len(train_loader.dataset) / train_loader.batch_size)

    print(f"Epoch: {epoch} \t Train Loss: {epoch_loss}")

    return epoch_loss


def test(model, device, test_loader, criterion):
    test_epoch_loss = 0

    model.eval()

    for batch_idx, (user, item, rating) in tqdm(enumerate(test_loader)):

        rating = rating.to(dtype=torch.float, device=device)
        user = user.to(dtype=torch.long, device=device)
        item = item.to(dtype=torch.long, device=device)

        # Predict and calculate loss
        prediction = model(user, item)
        loss = criterion(prediction, rating)

        test_epoch_loss += loss.item()

    test_epoch_loss /= int(len(test_loader.dataset) / test_loader.batch_size)

    print('\nTest Epoch loss: {:.4f}\n'.format(
        test_epoch_loss))

    return test_epoch_loss
