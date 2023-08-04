import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Initial transform to convert to tensor
transform = transforms.Compose([transforms.ToTensor()])

# Load STL10 dataset
dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)

# Prepare data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)

# Variables to store sum and squared sum of all pixels, and total count of pixels
mean = 0.
std = 0.
nb_samples = 0.

# Calculate mean
for data, _ in data_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples

# Calculate standard deviation
for data, _ in data_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    std += ((data - mean.unsqueeze(1)) ** 2).sum([0, 2])

std = torch.sqrt(std / (nb_samples * 96 * 96))

print('Mean: ', mean)
print('Std: ', std)
