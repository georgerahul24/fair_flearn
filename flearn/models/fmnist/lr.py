import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from flearn.utils.model_utils import batch_data, gen_batch


class LogisticRegression(nn.Module):
    def __init__(self, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(784, num_classes)  # 784 for 28x28 images
        # Add L2 regularization via weight_decay in optimizer
        
    def forward(self, x):
        return self.linear(x)


class Model(object):    
    def __init__(self, num_classes, q, optimizer, seed=1):
        
        # Set random seed for reproducibility
        torch.manual_seed(123 + seed)
        np.random.seed(123 + seed)
        
        # params
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # create model
        self.model = LogisticRegression(num_classes).to(self.device)
        
        # Store optimizer class and params (will be instantiated in create_optimizer)
        self.optimizer_class = optimizer
        self.optimizer = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Calculate model size and FLOPs
        self.size = self._calculate_model_size()
        self.flops = self._calculate_flops()
    
    def create_optimizer(self, lr=0.01, weight_decay=0.001):
        """Create optimizer with L2 regularization (weight_decay)"""
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
        """Approximate FLOPs calculation for linear layer"""
        # For a linear layer: FLOPs ≈ input_features * output_features * 2 (multiply + add)
        return 784 * self.num_classes * 2

    
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

    def get_gradients(self, data, model_len):
        """Get gradients for given data"""
        self.model.eval()
        
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
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

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
                grads.append(param.grad.data.cpu().numpy().flatten())
        flat_grads = np.concatenate(grads) if grads else np.array([])
        
        # Optimizer step
        self.optimizer.step()
        
        weights = self.get_params()
        return flat_grads, loss.item(), weights
    
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
