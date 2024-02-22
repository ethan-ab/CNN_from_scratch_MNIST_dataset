import torch
from torch import nn, optim
import model
from src.utils import dataloader


train_loader = dataloader.get_train_loader(batch_size=32)
val_loader = dataloader.get_val_loader(batch_size=32)
test_loader = dataloader.get_test_loader(batch_size=32)



for inputs, labels in train_loader:
    print("Batch inputs:", inputs.shape)
    print("Batch labels:", labels.shape)

model = model.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on validation set: {100 * correct / total}%")





