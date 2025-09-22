'''
This tutorial was taken from the PyTorch website.
It can take a minute or so to run and it will generate the model.pth file.
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

"""
This is just to help me understand what's going on:
The training_data variable is an object in the FashionMNIST class
It is an indexable object. Each index contains an image and a label (which is an integer corresponding to a bin/category)

first_training_image, first_training_label = training_data[0]
print(f"first_training_image data type: {first_training_image.dtype}")           # Output: first_training_data data type: torch.float32
print(f"first_training_image shape: {first_training_image.shape}")               # Output: first_training_data shape: torch.Size([1, 28, 28])
print(f"first_training_label data type: {type(first_training_label)}")           # Output: first_training_label data type: <class 'int'>
"""

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


"""
These dataloader objects are in the DataLoader class. The dataloader is not the batch itself, but it is the logic to make the batches.
Look at the creation of the objects. You'll see that all the data (training or testing) along with batch size must be passed into the object creation.
These dataloaders are similar to the training_data and test_data objects, but these are not indexable (they are only iterable)
But the dataloaders (like the datasets) come in image-label tuples.

for images_batch, labels_batch in train_dataloader:
    print("--- We got a batch! ---")
    print(f"Shape of the images batch: {images_batch.shape}")
    print(f"Shape of the labels batch: {labels_batch.shape}")
    
    # We only want to see the first batch, so we stop the loop
    break
"""

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

"""
# Creating Models

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# Optimizing the Model Parameters

loss_fn = nn.CrossEntropyLoss()             # for our purposes, we'd use a different error function. Perhaps nn.MSELoss() (Mean Squared Error). Or we could try nn.L1Loss() (just abs val error)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()               # put the model in training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)       # remember, X is the "questions" array and y is the "answers" array

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()         # calculates the gradients (how much each parameter contributed to the error)
        optimizer.step()        # let the optimizer update the parameters
        optimizer.zero_grad()   # reset the gradients to zero so they don't pile up in future batches

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()                # put the model in evaluation (testing) mode
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# Saving Models

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# Loading Models

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

    """