import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from ..models.phase_retrieval import PhaseCNN
from ..utils.dataset import PhaseRetrievalDataset
from .train_phase_retrieval import PhaseRetrievalTrainer

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train phase retrieval model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create datasets
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10)
    ])
    
    train_dataset = PhaseRetrievalDataset(
        data_dir=os.path.join(args.data_dir, 'train'),
        transform=transform
    )
    val_dataset = PhaseRetrievalDataset(
        data_dir=os.path.join(args.data_dir, 'val'),
        transform=None
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = PhaseCNN(
        input_size=(256, 256),
        n_channels=1,
        n_filters=32,
        n_layers=3
    )
    
    # Create trainer
    trainer = PhaseRetrievalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        config={
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'n_epochs': args.n_epochs,
            'save_dir': args.save_dir,
            'log_dir': args.log_dir,
            'save_freq': 10,
            'val_freq': 1,
            'early_stopping_patience': 10
        }
    )
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    main() 