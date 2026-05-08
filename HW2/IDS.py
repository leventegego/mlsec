import torch.nn as nn
import torch
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader


# Model architecture
class IntrusionDetector(nn.Module):
    """Simple feedforward neural network for intrusion detection."""
    
    def __init__(self, input_dim):
        super(IntrusionDetector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


# Training function
def train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=128, lr=0.001, device='cpu'):
    """
    Train the intrusion detection model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        model: Trained model
        train_losses: List of training losses per epoch
        test_accuracies: List of test accuracies per epoch
    """
    print("\n" + "="*70)
    print("Training Intrusion Detection Model")
    print("="*70)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = IntrusionDetector(input_dim)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    test_accuracies = []

    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = torch.sigmoid(model(X_test_tensor))
           
            test_predictions = (test_outputs > 0.5).float()
            test_accuracy = (test_predictions == y_test_tensor).float().mean().item()
            test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Training set
        train_outputs = torch.sigmoid(model(X_train_tensor))
        train_predictions = (train_outputs > 0.5).float()

        # Test set
        test_outputs = torch.sigmoid(model(X_test_tensor))
        test_predictions = (test_outputs > 0.5).float()
   
        print("\n" + "="*70)
    print("Final Model Performance")
    print("="*70)

    # compute train accuracy with sckit-learn for more detailed metrics
    
    # print training accuracy
    print("\nClassification Report (Training Set):")
    print(classification_report(y_train_tensor.cpu(), train_predictions.cpu(), target_names=['Normal', 'Attack']))

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test_tensor.cpu(), test_predictions.cpu(), target_names=['Normal', 'Attack']))

    return model