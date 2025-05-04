import torch
import pytest
import numpy as np
from fourierlab.ml.pinn import PINN, PINNTrainer
from torch.utils.data import DataLoader, TensorDataset

@pytest.fixture
def sample_field():
    """Generate sample complex field"""
    size = (1, 1, 32, 32)  # (batch, channel, height, width)
    real = torch.randn(size)
    imag = torch.randn(size)
    return real + 1j * imag

@pytest.fixture
def sample_distance():
    """Generate sample propagation distance"""
    return torch.tensor([0.1])  # 0.1 meters

@pytest.fixture
def sample_dataset(sample_field, sample_distance):
    """Create sample dataset"""
    dataset = TensorDataset(
        sample_field.real,
        sample_field.imag,
        sample_distance
    )
    return DataLoader(dataset, batch_size=1)

def test_pinn_initialization():
    """Test PINN initialization"""
    model = PINN(
        input_size=(32, 32),
        hidden_dims=[32, 64, 32],
        wavelength=632.8e-9,
        pixel_size=5e-6
    )
    
    assert model.input_size == (32, 32)
    assert model.hidden_dims == [32, 64, 32]
    assert model.wavelength == 632.8e-9
    assert model.pixel_size == 5e-6
    assert len(model.layers) == 3

def test_pinn_forward(sample_field):
    """Test PINN forward pass"""
    model = PINN(input_size=(32, 32))
    predicted_field, physics_losses = model(sample_field)
    
    assert predicted_field.shape == sample_field.shape
    assert isinstance(physics_losses, dict)
    assert all(k in physics_losses for k in [
        'helmholtz', 'energy', 'phase_continuity'
    ])
    assert all(isinstance(v, torch.Tensor) for v in physics_losses.values())
    assert all(v.ndim == 0 for v in physics_losses.values())  # Scalar losses

def test_pinn_physics_losses(sample_field):
    """Test physics-based losses"""
    model = PINN(input_size=(32, 32))
    _, physics_losses = model(sample_field)
    
    # Check Helmholtz loss
    assert physics_losses['helmholtz'] >= 0
    assert not torch.isnan(physics_losses['helmholtz'])
    
    # Check energy conservation loss
    assert physics_losses['energy'] >= 0
    assert not torch.isnan(physics_losses['energy'])
    
    # Check phase continuity loss
    assert physics_losses['phase_continuity'] >= 0
    assert not torch.isnan(physics_losses['phase_continuity'])

def test_pinn_propagation_loss(sample_field, sample_distance):
    """Test propagation loss"""
    model = PINN(input_size=(32, 32))
    _, physics_losses = model(sample_field, sample_distance)
    
    assert 'propagation' in physics_losses
    assert physics_losses['propagation'] >= 0
    assert not torch.isnan(physics_losses['propagation'])

def test_pinn_trainer_initialization(sample_dataset):
    """Test PINN trainer initialization"""
    model = PINN(input_size=(32, 32))
    trainer = PINNTrainer(
        model=model,
        train_loader=sample_dataset,
        config={
            'learning_rate': 1e-4,
            'helmholtz_weight': 1.0,
            'energy_weight': 0.1,
            'phase_continuity_weight': 0.1,
            'propagation_weight': 1.0
        }
    )
    
    assert trainer.model == model
    assert trainer.train_loader == sample_dataset
    assert trainer.config['learning_rate'] == 1e-4
    assert all(k in trainer.loss_weights for k in [
        'helmholtz', 'energy', 'phase_continuity', 'propagation'
    ])

def test_pinn_trainer_train_epoch(sample_dataset):
    """Test PINN trainer training epoch"""
    model = PINN(input_size=(32, 32))
    trainer = PINNTrainer(model=model, train_loader=sample_dataset)
    
    loss = trainer.train_epoch()
    assert isinstance(loss, float)
    assert loss >= 0
    assert not np.isnan(loss)

def test_pinn_trainer_validate(sample_dataset):
    """Test PINN trainer validation"""
    model = PINN(input_size=(32, 32))
    trainer = PINNTrainer(
        model=model,
        train_loader=sample_dataset,
        val_loader=sample_dataset
    )
    
    val_loss = trainer.validate()
    assert isinstance(val_loss, float)
    assert val_loss >= 0
    assert not np.isnan(val_loss)

def test_pinn_trainer_full_training(sample_dataset):
    """Test full PINN training process"""
    model = PINN(input_size=(32, 32))
    trainer = PINNTrainer(
        model=model,
        train_loader=sample_dataset,
        val_loader=sample_dataset
    )
    
    metrics = trainer.train(num_epochs=2)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['train_loss', 'val_loss'])
    assert len(metrics['train_loss']) == 2
    assert len(metrics['val_loss']) == 2
    assert all(isinstance(v, float) for v in metrics['train_loss'])
    assert all(isinstance(v, float) for v in metrics['val_loss'])

def test_pinn_gradient_flow(sample_field):
    """Test gradient flow in PINN"""
    model = PINN(input_size=(32, 32))
    predicted_field, physics_losses = model(sample_field)
    
    # Check if gradients flow properly
    total_loss = sum(physics_losses.values())
    total_loss.backward()
    
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()

def test_pinn_different_input_sizes():
    """Test PINN with different input sizes"""
    sizes = [(16, 16), (32, 32), (64, 64)]
    
    for size in sizes:
        model = PINN(input_size=size)
        field = torch.randn(1, 1, *size) + 1j * torch.randn(1, 1, *size)
        predicted_field, _ = model(field)
        
        assert predicted_field.shape == field.shape

def test_pinn_different_hidden_dims():
    """Test PINN with different hidden dimensions"""
    hidden_dims_list = [
        [32],
        [32, 64],
        [32, 64, 128, 64, 32]
    ]
    
    for hidden_dims in hidden_dims_list:
        model = PINN(
            input_size=(32, 32),
            hidden_dims=hidden_dims
        )
        field = torch.randn(1, 1, 32, 32) + 1j * torch.randn(1, 1, 32, 32)
        predicted_field, _ = model(field)
        
        assert predicted_field.shape == field.shape

def test_pinn_physics_constraints():
    """Test physics constraints in PINN"""
    model = PINN(input_size=(32, 32))
    field = torch.randn(1, 1, 32, 32) + 1j * torch.randn(1, 1, 32, 32)
    _, physics_losses = model(field)
    
    # Check if physics losses decrease with training
    optimizer = torch.optim.Adam(model.parameters())
    
    initial_losses = {k: v.item() for k, v in physics_losses.items()}
    
    for _ in range(10):
        optimizer.zero_grad()
        _, physics_losses = model(field)
        total_loss = sum(physics_losses.values())
        total_loss.backward()
        optimizer.step()
    
    final_losses = {k: v.item() for k, v in physics_losses.items()}
    
    # Check if losses decreased
    for k in initial_losses:
        assert final_losses[k] <= initial_losses[k]

def test_pinn_save_load():
    """Test PINN model saving and loading"""
    # Create and save model
    model1 = PINN(input_size=(32, 32))
    torch.save(model1.state_dict(), 'test_pinn.pt')
    
    # Load model
    model2 = PINN(input_size=(32, 32))
    model2.load_state_dict(torch.load('test_pinn.pt'))
    
    # Compare models
    field = torch.randn(1, 1, 32, 32) + 1j * torch.randn(1, 1, 32, 32)
    out1, _ = model1(field)
    out2, _ = model2(field)
    
    assert torch.allclose(out1, out2) 