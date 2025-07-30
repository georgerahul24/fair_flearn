import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from flearn.utils.model_utils import batch_data


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense layers
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Reshape input from [batch, 784] to [batch, 1, 28, 28]
        x = x.view(-1, 1, 28, 28)
        
        # Conv layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for dense layers
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class Model(object):
    '''
    Assumes that images are 28px by 28px
    '''

    def __init__(self, num_classes, q, optimizer, seed=1):
        
        # Set random seed for reproducibility
        torch.manual_seed(123 + seed)
        np.random.seed(123 + seed)
        
        # params
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # create model
        self.model = CNN(num_classes).to(self.device)
        
        # Store optimizer class and params (will be instantiated in create_optimizer)
        self.optimizer_class = optimizer
        self.optimizer = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Calculate model size and FLOPs
        self.size = self._calculate_model_size()
        self.flops = self._calculate_flops()
    
    def create_optimizer(self, lr=0.01, weight_decay=0.0):
        """Create optimizer"""
        if self.optimizer_class == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer_class == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            # Default to SGD if optimizer not recognized
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _calculate_model_size(self):
        """Calculate model size in bytes"""
        total_size = 0
        for param in self.model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def _calculate_flops(self):
        """Approximate FLOPs calculation for CNN"""
        # This is a simplified calculation - in practice you'd want more precise FLOP counting
        # Conv1: 28*28*32*5*5*1 + Conv2: 14*14*64*5*5*32 + FC1: 7*7*64*2048 + FC2: 2048*num_classes
        conv1_flops = 28 * 28 * 32 * 5 * 5 * 1
        conv2_flops = 14 * 14 * 64 * 5 * 5 * 32
        fc1_flops = 7 * 7 * 64 * 2048
        fc2_flops = 2048 * self.num_classes
        return conv1_flops + conv2_flops + fc1_flops + fc2_flops

    def set_params(self, model_params=None):
        """Set model parameters"""
        if model_params is not None:
            param_dict = {}
            param_names = []
            for name, param in self.model.named_parameters():
                param_names.append(name)
            
            # Assuming model_params is a list of numpy arrays in the same order
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if i < len(model_params):
                    param.data = torch.from_numpy(model_params[i]).to(self.device)

    def get_params(self):
        """Get model parameters as list of numpy arrays"""
        model_params = []
        for param in self.model.parameters():
            model_params.append(param.data.cpu().numpy())
        return model_params

    def get_gradients(self, data, latest_model):
        """Get gradients for given data"""
        self.model.eval()
        
        # Calculate model length from latest_model (list of parameter arrays)
        model_len = sum(param.size for param in latest_model)
        
        # Convert data to tensors
        features = torch.from_numpy(data['x']).float().to(self.device)
        labels = torch.from_numpy(data['y']).long().to(self.device)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Extract gradients and flatten
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.cpu().numpy().flatten())
        
        # Concatenate all gradients
        flat_grads = np.concatenate(grads) if grads else np.zeros(model_len)
        
        num_samples = len(data['y'])
        return num_samples, flat_grads

    def get_loss(self, data):
        """Get loss for given data"""
        self.model.eval()
        
        # Convert data to tensors
        features = torch.from_numpy(data['x']).float().to(self.device)
        labels = torch.from_numpy(data['y']).long().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
        
        return loss.item()

    def solve_sgd(self, mini_batch_data):
        """Single SGD step"""
        if self.optimizer is None:
            self.create_optimizer()
            
        self.model.train()
        
        # Convert data to tensors
        features = torch.from_numpy(mini_batch_data[0]).float().to(self.device)
        labels = torch.from_numpy(mini_batch_data[1]).long().to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Extract gradients before optimizer step
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.cpu().numpy())
        
        # Optimizer step
        self.optimizer.step()
        
        weights = self.get_params()
        return grads, loss.item(), weights

    def solve_inner(self, data, num_epochs=1, batch_size=32):
        """Solves local optimization problem"""
        if self.optimizer is None:
            self.create_optimizer()
            
        self.model.train()
        
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            for X, y in batch_data(data, batch_size):
                # Convert to tensors
                features = torch.from_numpy(X).float().to(self.device)
                labels = torch.from_numpy(y).long().to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
        
        soln = self.get_params()
        comp = num_epochs * (len(data['y']) // batch_size) * batch_size * self.flops
        return soln, comp

    def test(self, data):
        """Test the model
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        """
        self.model.eval()
        
        # Convert data to tensors
        features = torch.from_numpy(data['x']).float().to(self.device)
        labels = torch.from_numpy(data['y']).long().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            tot_correct = (predicted == labels).sum().item()
        
        return tot_correct, loss.item()

    def close(self):
        """Close/cleanup - not needed in PyTorch but kept for compatibility"""
        pass
