import torch
from torch.utils.data import Dataset,ConcatDataset
from torchvision import datasets,transforms
import gc

class MNISTDataLoader:
    def __init__(self, batch_size=128, seed=1):
        self.cuda = torch.cuda.is_available()
        self.batch_size = batch_size
        torch.manual_seed(seed)
        
        # Calculate mean and std
        self.mean, self.std = self._calculate_stats()
        
        # Update transforms with calculated values
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((self.mean,), (self.std,)),
            transforms.Resize((28, 28)),
            transforms.RandomRotation(10),
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))
        ])
        
        self.dataloader_args = dict(
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        ) if self.cuda else dict(shuffle=True, batch_size=64)
        

    def _calculate_stats(self):
        # Load entire dataset in one batch
        temp_data1 = datasets.MNIST(
            root='.data',
            train=True,
            download=True,
            transform=transforms.ToTensor()  # Only convert to tensor without normalization
        )
        # Load test dataset
        temp_data2 = datasets.MNIST(
            root='.data',
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        combined_dataset = ConcatDataset([temp_data1, temp_data2])

        loader1 = torch.utils.data.DataLoader(combined_dataset, batch_size=len(combined_dataset), shuffle=False)
        data1 = next(iter(loader1))[0]

        # Calculate mean and std
        mean = data1.mean().item()
        std = data1.std().item()

        # Clean up
        del temp_data1, temp_data2, loader1, data1
        gc.collect()

        return mean, std

    def get_data_loaders(self):
        train_data = datasets.MNIST(
            root='.data',
            train=True,
            download=True,
            transform=self.train_transforms
        )
        
        test_data = datasets.MNIST(
            root='data',
            train=False,
            download=True,
            transform=self.test_transforms
        )
        
        train_loader = torch.utils.data.DataLoader(train_data, **self.dataloader_args)
        test_loader = torch.utils.data.DataLoader(test_data, **self.dataloader_args)
        
        return train_loader, test_loader

if __name__ == '__main__':
    data_loader = MNISTDataLoader()
    train_loader, test_loader = data_loader.get_data_loaders()
    data_loader.mean, data_loader.std
    print(len(train_loader.dataset), len(test_loader.dataset))
    