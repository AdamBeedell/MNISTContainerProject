## MNIST one

## installed torch, torchvision, torchaudio (probably dont need that last one but it came reccomended)
## that has come with mpmath, typing-extensions, sympy, setuptools, pillow, numpy, networkx, MarkupSafe, fsspec, filelock, jinja2
## as dependencies


## ip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 
## index url needed to enable CUDA handling

import torch
print(torch.__version__)
print(torch.cuda.is_available())  ## looking for True
import torchvision
import torchvision.transforms as transforms



### Making a pipeline to transform images from 0-255 intensity values to -1 to 1 representing the same thing
## the pipeline is not exactly a pytorch specific thing in this format, but is module-specific
## the transform applies only in-memory and only when fetching the items (ie, on the way to training or testing)


bwminus1to1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=bwminus1to1, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=bwminus1to1, download=True)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim


class MNISTModel(NN.Module):   ### This creates a class for our specific NN, inheriting from the pytorch equivalent
    def __init__(self):  
        super().__init__()  ## super goes up one level to the torch NN module, and initializes the net
        self.fc1 = NN.Linear(28 * 28, 256)  # First hidden layer (784 pixel slots, gradually reducing down)
        self.fc2 = NN.Linear(256, 128)  # half as many nodes
        self.fc3 = NN.Linear(128, 64)   # half as many nodes
        self.fc4 = NN.Linear(64, 10) # Output layer (64 -> 10, one for each valid prediction)

    def forward(self, x):  # feed forward
        x = x.view(-1, 28 * 28)  # Flatten input from (batch, 1, 28, 28) -> (batch, 784), applies to the tensor prepared above in the dataloader
        x = F.relu(self.fc1(x))  # Activation function (ReLU), no negatives, play with leaky ReLU later
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here, end of the road ("cross-entropy expects raw logits" - which are produced here, the logits will be converted to probabilities later by the cross-entropy function during training and softmax during training and inference)
        return x
    
loss_function = NN.CrossEntropyLoss()


model = MNISTModel()

optimizer = optim.Adam(model.parameters(), lr=0.001)