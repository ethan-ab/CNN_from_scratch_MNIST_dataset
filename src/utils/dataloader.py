from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.utils import load_data

def get_train_loader(batch_size):
    train_loader = DataLoader(load_data.train_set, batch_size=batch_size, shuffle=True)
    return train_loader

train_loader = get_train_loader(32)
def get_val_loader(batch_size):
    val_loader = DataLoader(load_data.val_set, batch_size=batch_size, shuffle=True)
    return val_loader

def get_test_loader(batch_size):
    test_loader = DataLoader(load_data.test_set, batch_size=batch_size, shuffle=True)
    return test_loader
