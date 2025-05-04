import torch
import torch.nn as nn
import pytest
from fourierlab.ml.models import FourierCNN, FourierUNet
from fourierlab.ml.transformer_models import FourierTransformer, FourierTransformerUNet
from fourierlab.ml.losses import FourierLoss, PhaseMaskLoss, SSIMLoss
from fourierlab.ml.augmentations import FourierAugmentation
from fourierlab.ml.optimizers import (
    FourierOptimizer,
    FourierGradientClipping,
    FourierLearningRateFinder,
    FourierWeightDecay
)

@pytest.fixture
def sample_input():
    return torch.randn(2, 1, 32, 32)

@pytest.fixture
def sample_target():
    return torch.randn(2, 1, 32, 32)

def test_fourier_cnn(sample_input):
    model = FourierCNN(in_channels=1, num_classes=2)
    output = model(sample_input)
    assert output.shape == (2, 2)
    assert not torch.isnan(output).any()

def test_fourier_unet(sample_input):
    model = FourierUNet(in_channels=1, out_channels=1)
    output = model(sample_input)
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()

def test_fourier_transformer(sample_input):
    model = FourierTransformer(in_channels=1)
    output = model(sample_input)
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()

def test_fourier_transformer_unet(sample_input):
    model = FourierTransformerUNet(in_channels=1, out_channels=1)
    output = model(sample_input)
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()

def test_fourier_loss(sample_input, sample_target):
    criterion = FourierLoss()
    loss = criterion(sample_input, sample_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

def test_phase_mask_loss(sample_input, sample_target):
    criterion = PhaseMaskLoss()
    phase_mask = torch.randn_like(sample_input)
    loss, metrics = criterion(sample_input, sample_target, phase_mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["reconstruction_loss", "smoothness_loss", "contrast_loss"])

def test_ssim_loss(sample_input, sample_target):
    criterion = SSIMLoss()
    loss = criterion(sample_input, sample_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)

def test_fourier_augmentation(sample_input):
    aug = FourierAugmentation()
    augmented = aug.apply_augmentation(sample_input)
    assert augmented.shape == sample_input.shape
    assert not torch.isnan(augmented).any()

def test_fourier_optimizer(sample_input, sample_target):
    model = FourierCNN(in_channels=1, num_classes=2)
    optimizer = FourierOptimizer(model)
    criterion = nn.CrossEntropyLoss()
    
    # Forward pass
    output = model(sample_input)
    loss = criterion(output, sample_target.argmax(dim=1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check learning rate
    lr = optimizer.get_lr()
    assert isinstance(lr, float)
    assert lr > 0

def test_fourier_gradient_clipping(sample_input, sample_target):
    model = FourierCNN(in_channels=1, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    clipper = FourierGradientClipping()
    
    # Forward pass
    output = model(sample_input)
    loss = criterion(output, sample_target.argmax(dim=1))
    
    # Backward pass
    loss.backward()
    
    # Clip gradients
    clipper.clip_gradients(model)
    
    # Check gradients
    for param in model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            assert torch.norm(param.grad) <= clipper.max_norm

def test_fourier_weight_decay(sample_input):
    model = FourierCNN(in_channels=1, num_classes=2)
    weight_decay = FourierWeightDecay(model)
    
    # Compute regularization
    reg_loss = weight_decay.compute_regularization()
    assert isinstance(reg_loss, torch.Tensor)
    assert reg_loss.ndim == 0
    assert not torch.isnan(reg_loss)

def test_model_initialization():
    # Test different model configurations
    configs = [
        {"in_channels": 1, "num_classes": 2},
        {"in_channels": 3, "num_classes": 10},
        {"in_channels": 1, "num_classes": 2, "base_channels": 64},
        {"in_channels": 1, "num_classes": 2, "num_blocks": 4}
    ]
    
    for config in configs:
        model = FourierCNN(**config)
        x = torch.randn(2, config["in_channels"], 32, 32)
        output = model(x)
        assert output.shape == (2, config["num_classes"])
        assert not torch.isnan(output).any()

def test_loss_functions():
    # Test different loss configurations
    x = torch.randn(2, 1, 32, 32)
    y = torch.randn(2, 1, 32, 32)
    
    # FourierLoss
    for alpha, beta, gamma in [(0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.3, 0.4, 0.3)]:
        criterion = FourierLoss(alpha=alpha, beta=beta, gamma=gamma)
        loss = criterion(x, y)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
    
    # PhaseMaskLoss
    phase_mask = torch.randn_like(x)
    for smoothness, contrast in [(0.1, 0.05), (0.2, 0.1), (0.05, 0.02)]:
        criterion = PhaseMaskLoss(
            smoothness_weight=smoothness,
            contrast_weight=contrast
        )
        loss, metrics = criterion(x, y, phase_mask)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert isinstance(metrics, dict)
    
    # SSIMLoss
    for window_size in [11, 7, 15]:
        criterion = SSIMLoss(window_size=window_size)
        loss = criterion(x, y)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

def test_augmentation_combinations():
    x = torch.randn(2, 1, 32, 32)
    aug = FourierAugmentation()
    
    # Test different augmentation combinations
    combinations = [
        {"phase_noise": True, "magnitude_noise": False, "rotation": False, "scale": False, "flip": False},
        {"phase_noise": False, "magnitude_noise": True, "rotation": False, "scale": False, "flip": False},
        {"phase_noise": False, "magnitude_noise": False, "rotation": True, "scale": False, "flip": False},
        {"phase_noise": False, "magnitude_noise": False, "rotation": False, "scale": True, "flip": False},
        {"phase_noise": False, "magnitude_noise": False, "rotation": False, "scale": False, "flip": True},
        {"phase_noise": True, "magnitude_noise": True, "rotation": True, "scale": True, "flip": True}
    ]
    
    for combo in combinations:
        augmented = aug.apply_augmentation(x, **combo)
        assert augmented.shape == x.shape
        assert not torch.isnan(augmented).any()

def test_optimizer_configurations():
    model = FourierCNN(in_channels=1, num_classes=2)
    
    # Test different optimizer configurations
    configs = [
        {"lr": 1e-3, "weight_decay": 1e-4},
        {"lr": 1e-4, "weight_decay": 1e-5},
        {"lr": 1e-2, "weight_decay": 1e-3}
    ]
    
    for config in configs:
        optimizer = FourierOptimizer(model, **config)
        assert optimizer.get_lr() == config["lr"]
        
        # Test step
        x = torch.randn(2, 1, 32, 32)
        y = torch.randn(2, 2)
        criterion = nn.CrossEntropyLoss()
        
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert not any(torch.isnan(p).any() for p in model.parameters()) 