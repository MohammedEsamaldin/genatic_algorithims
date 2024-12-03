# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt


train_set = torchvision.datasets.FashionMNIST(root = ".", train=True,
                                              download=False,
                                              transform=transforms.ToTensor())
validation_set = torchvision.datasets.FashionMNIST(root = ".", train=False,
                                             download=False,
                                             transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False)
# Fix the seed to be able to get the same randomness across runs and
# hence reproducible outcomes
torch.manual_seed(0)


# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=1024)  # Flattened size = 64 * 4 * 4
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.relu4 = nn.ReLU()
        self.output = nn.Linear(in_features=256, out_features=10)
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.output(x)
        
        return x
def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

model = MyCNN().to(device)
initialize_weights(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(model.parameters()), lr=0.1)

# Training settings
epochs = 30
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    train_loss = 0
    train_correct = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        train_loss += loss.item()  # Accumulate loss
        _, predicted = torch.max(output, 1)  # Get predictions
        train_correct += (predicted == target).sum().item()
    
    # Calculate training accuracy
    train_accuracy = train_correct / len(train_loader.dataset)
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    
    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    val_correct = 0
    with torch.no_grad():  # No need to compute gradients
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            val_correct += (predicted == target).sum().item()
    
    val_accuracy = val_correct / len(validation_loader.dataset)
    val_losses.append(val_loss / len(validation_loader))
    val_accuracies.append(val_accuracy)
    
    # Print metrics for the epoch
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the best model
torch.save(model.state_dict(), "best_cnn.pth")
print("Model saved as best_cnn.pth")

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()