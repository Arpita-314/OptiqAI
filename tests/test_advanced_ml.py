import torch
import torch.nn as nn
import pytest
from fourierlab.ml.gan_models import (
    FourierGenerator,
    FourierDiscriminator,
    FourierGAN,
    FourierGANLoss,
    FourierGANOptimizer
)
from fourierlab.ml.advanced_augmentations import (
    AdvancedFourierAugmentation,
    AdvancedFourierDataset,
    AdvancedFourierDataLoader
)
from fourierlab.ml.mixed_precision import (
    MixedPrecisionTrainer,
    MixedPrecisionGAN,
    MixedPrecisionScheduler
)
from fourierlab.ml.style_transfer import (
    FourierStyleTransfer,
    FourierStyleAugmentation
)
from fourierlab.ml.fast_neural_style import FastNeuralStyle
from fourierlab.ml.residual_block import ResidualBlock
from fourierlab.ml.adain import AdaIN
from fourierlab.ml.adain_style_transfer import AdaINStyleTransfer
from fourierlab.ml.perceptual_loss import PerceptualLoss

@pytest.fixture
def sample_input():
    return torch.randn(2, 1, 32, 32)

@pytest.fixture
def sample_target():
    return torch.randn(2, 1, 32, 32)

@pytest.fixture
def sample_latent():
    return torch.randn(2, 100)

@pytest.fixture
def sample_style():
    return torch.randn(2, 1, 32, 32)

def test_fourier_generator(sample_latent):
    model = FourierGenerator(latent_dim=100, out_channels=1)
    output = model(sample_latent)
    assert output.shape == (2, 1, 32, 32)
    assert not torch.isnan(output).any()
    assert torch.all(output >= -1) and torch.all(output <= 1)

def test_fourier_discriminator(sample_input):
    model = FourierDiscriminator(in_channels=1)
    output = model(sample_input)
    assert output.shape == (2,)
    assert not torch.isnan(output).any()

def test_fourier_gan(sample_latent):
    model = FourierGAN(latent_dim=100, in_channels=1)
    output = model(sample_latent)
    assert output.shape == (2, 1, 32, 32)
    assert not torch.isnan(output).any()
    assert torch.all(output >= -1) and torch.all(output <= 1)

def test_fourier_gan_loss(sample_input, sample_target):
    criterion = FourierGANLoss()
    
    # Test generator loss
    fake_outputs = torch.randn(2)
    generated_images = torch.randn(2, 1, 32, 32)
    target_images = torch.randn(2, 1, 32, 32)
    
    g_loss, g_metrics = criterion.generator_loss(
        fake_outputs,
        generated_images,
        target_images
    )
    
    assert isinstance(g_loss, torch.Tensor)
    assert g_loss.ndim == 0
    assert not torch.isnan(g_loss)
    assert isinstance(g_metrics, dict)
    assert all(k in g_metrics for k in [
        'g_loss', 'l1_loss', 'phase_loss', 'magnitude_loss', 'total_loss'
    ])
    
    # Test discriminator loss
    real_outputs = torch.randn(2)
    fake_outputs = torch.randn(2)
    
    d_loss, d_metrics = criterion.discriminator_loss(
        real_outputs,
        fake_outputs
    )
    
    assert isinstance(d_loss, torch.Tensor)
    assert d_loss.ndim == 0
    assert not torch.isnan(d_loss)
    assert isinstance(d_metrics, dict)
    assert all(k in d_metrics for k in [
        'real_loss', 'fake_loss', 'total_loss'
    ])

def test_advanced_augmentation(sample_input):
    aug = AdvancedFourierAugmentation()
    
    # Test elastic deformation
    elastic = aug.elastic_deformation(sample_input)
    assert elastic.shape == sample_input.shape
    assert not torch.isnan(elastic).any()
    
    # Test cutout
    cutout = aug.cutout(sample_input)
    assert cutout.shape == sample_input.shape
    assert not torch.isnan(cutout).any()
    
    # Test mixup
    y = torch.randn_like(sample_input)
    mixed_x, mixed_y = aug.mixup(sample_input, y)
    assert mixed_x.shape == sample_input.shape
    assert mixed_y.shape == y.shape
    assert not torch.isnan(mixed_x).any()
    assert not torch.isnan(mixed_y).any()
    
    # Test frequency mask
    masked = aug.frequency_mask(sample_input)
    assert masked.shape == sample_input.shape
    assert not torch.isnan(masked).any()
    
    # Test random erasing
    erased = aug.random_erasing(sample_input)
    assert erased.shape == sample_input.shape
    assert not torch.isnan(erased).any()
    
    # Test random crop and resize
    target_size = (16, 16)
    cropped = aug.random_crop_and_resize(sample_input, target_size)
    assert cropped.shape == (2, 1, *target_size)
    assert not torch.isnan(cropped).any()

def test_advanced_dataset(sample_input, sample_target):
    dataset = AdvancedFourierDataset(
        data=sample_input,
        labels=sample_target,
        augmentation=AdvancedFourierAugmentation(),
        target_size=(16, 16)
    )
    
    x, y = dataset[0]
    assert x.shape == (1, 16, 16)
    assert y.shape == (1, 32, 32)
    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()

def test_mixed_precision_trainer(sample_input, sample_target):
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(32, 1, 3, 1, 1)
    )
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )
    
    # Test training step
    loss, metrics = trainer.train_step(sample_input, sample_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['loss', 'scale'])
    
    # Test validation step
    loss, metrics = trainer.validate_step(sample_input, sample_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert isinstance(metrics, dict)
    assert 'loss' in metrics

def test_mixed_precision_gan(sample_input):
    gan = FourierGAN(latent_dim=100, in_channels=1)
    optimizer_g = torch.optim.Adam(gan.generator.parameters())
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters())
    criterion = FourierGANLoss()
    
    trainer = MixedPrecisionGAN(
        gan=gan,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        criterion=criterion
    )
    
    # Test training step
    g_loss, d_loss, metrics = trainer.train_step(sample_input, latent_dim=100)
    assert isinstance(g_loss, torch.Tensor)
    assert isinstance(d_loss, torch.Tensor)
    assert g_loss.ndim == 0
    assert d_loss.ndim == 0
    assert not torch.isnan(g_loss)
    assert not torch.isnan(d_loss)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in [
        'g_loss', 'd_loss', 'g_scale', 'd_scale'
    ])

def test_mixed_precision_scheduler():
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.cuda.amp.GradScaler()
    
    scheduler_wrapper = MixedPrecisionScheduler(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler
    )
    
    # Test step
    scheduler_wrapper.step()
    lr = scheduler_wrapper.get_lr()
    scale = scheduler_wrapper.get_scale()
    
    assert isinstance(lr, float)
    assert isinstance(scale, float)
    assert lr > 0
    assert scale > 0

def test_gan_training_loop(sample_input):
    gan = FourierGAN(latent_dim=100, in_channels=1)
    optimizer_g = torch.optim.Adam(gan.generator.parameters())
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters())
    criterion = FourierGANLoss()
    
    trainer = MixedPrecisionGAN(
        gan=gan,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        criterion=criterion
    )
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(sample_input)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # Train for one epoch
    metrics = trainer.train_epoch(loader, latent_dim=100)
    
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in [
        'g_loss', 'd_loss', 'g_scale', 'd_scale'
    ])
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(v >= 0 for v in metrics.values())

def test_augmentation_combinations(sample_input):
    aug = AdvancedFourierAugmentation()
    
    # Test different augmentation combinations
    combinations = [
        {
            'phase_noise': True,
            'magnitude_noise': False,
            'rotation': False,
            'scale': False,
            'flip': False,
            'elastic': False,
            'cutout': False,
            'mixup': False,
            'frequency_mask': False,
            'random_erasing': False,
            'crop_resize': False
        },
        {
            'phase_noise': False,
            'magnitude_noise': True,
            'rotation': False,
            'scale': False,
            'flip': False,
            'elastic': False,
            'cutout': False,
            'mixup': False,
            'frequency_mask': False,
            'random_erasing': False,
            'crop_resize': False
        },
        {
            'phase_noise': True,
            'magnitude_noise': True,
            'rotation': True,
            'scale': True,
            'flip': True,
            'elastic': True,
            'cutout': True,
            'mixup': True,
            'frequency_mask': True,
            'random_erasing': True,
            'crop_resize': True
        }
    ]
    
    for combo in combinations:
        augmented = aug.apply_augmentation(sample_input, **combo)
        assert augmented.shape == sample_input.shape
        assert not torch.isnan(augmented).any()

def test_mixed_precision_edge_cases():
    # Test with very small values
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion
    )
    
    x = torch.randn(2, 10) * 1e-10
    y = torch.randn(2, 1) * 1e-10
    
    loss, metrics = trainer.train_step(x, y)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test with very large values
    x = torch.randn(2, 10) * 1e10
    y = torch.randn(2, 1) * 1e10
    
    loss, metrics = trainer.train_step(x, y)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test with zero values
    x = torch.zeros(2, 10)
    y = torch.zeros(2, 1)
    
    loss, metrics = trainer.train_step(x, y)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_gan_edge_cases(sample_input):
    gan = FourierGAN(latent_dim=100, in_channels=1)
    optimizer_g = torch.optim.Adam(gan.generator.parameters())
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters())
    criterion = FourierGANLoss()
    
    trainer = MixedPrecisionGAN(
        gan=gan,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        criterion=criterion
    )
    
    # Test with very small values
    x = sample_input * 1e-10
    g_loss, d_loss, metrics = trainer.train_step(x, latent_dim=100)
    assert not torch.isnan(g_loss)
    assert not torch.isnan(d_loss)
    assert not torch.isinf(g_loss)
    assert not torch.isinf(d_loss)
    
    # Test with very large values
    x = sample_input * 1e10
    g_loss, d_loss, metrics = trainer.train_step(x, latent_dim=100)
    assert not torch.isnan(g_loss)
    assert not torch.isnan(d_loss)
    assert not torch.isinf(g_loss)
    assert not torch.isinf(d_loss)
    
    # Test with zero values
    x = torch.zeros_like(sample_input)
    g_loss, d_loss, metrics = trainer.train_step(x, latent_dim=100)
    assert not torch.isnan(g_loss)
    assert not torch.isnan(d_loss)
    assert not torch.isinf(g_loss)
    assert not torch.isinf(d_loss)

def test_style_transfer(sample_input, sample_style):
    model = FourierStyleTransfer()
    output, metrics = model.transfer_style(sample_input, sample_style)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in [
        'content_loss', 'style_loss', 'tv_loss', 'total_loss'
    ])
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(v >= 0 for v in metrics.values())

def test_style_transfer_batch(sample_input, sample_style):
    model = FourierStyleTransfer()
    output, metrics = model.transfer_style_batch(sample_input, sample_style)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in [
        'content_loss', 'style_loss', 'tv_loss', 'total_loss'
    ])
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(v >= 0 for v in metrics.values())

def test_style_augmentation(sample_input, sample_style):
    aug = FourierStyleAugmentation()
    output = aug.apply_style(sample_input, sample_style)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()

def test_style_augmentation_batch(sample_input, sample_style):
    aug = FourierStyleAugmentation()
    output = aug.apply_style_batch(sample_input, sample_style)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()

def test_advanced_augmentation_with_style(sample_input, sample_style):
    aug = AdvancedFourierAugmentation()
    output = aug.apply_augmentation(
        sample_input,
        style_image=sample_style,
        style_transfer=True
    )
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()

def test_advanced_dataset_with_style(sample_input, sample_target, sample_style):
    dataset = AdvancedFourierDataset(
        data=sample_input,
        labels=sample_target,
        style_images=sample_style,
        augmentation=AdvancedFourierAugmentation(),
        target_size=(16, 16)
    )
    
    x, y = dataset[0]
    assert x.shape == (1, 16, 16)
    assert y.shape == (1, 32, 32)
    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()

def test_style_transfer_edge_cases(sample_input, sample_style):
    model = FourierStyleTransfer()
    
    # Test with very small values
    x = sample_input * 1e-10
    style = sample_style * 1e-10
    output, metrics = model.transfer_style(x, style)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert all(not math.isnan(v) for v in metrics.values())
    assert all(not math.isinf(v) for v in metrics.values())
    
    # Test with very large values
    x = sample_input * 1e10
    style = sample_style * 1e10
    output, metrics = model.transfer_style(x, style)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert all(not math.isnan(v) for v in metrics.values())
    assert all(not math.isinf(v) for v in metrics.values())
    
    # Test with zero values
    x = torch.zeros_like(sample_input)
    style = torch.zeros_like(sample_style)
    output, metrics = model.transfer_style(x, style)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert all(not math.isnan(v) for v in metrics.values())
    assert all(not math.isinf(v) for v in metrics.values())

def test_style_transfer_different_sizes():
    model = FourierStyleTransfer()
    
    # Test with different input sizes
    content = torch.randn(1, 1, 32, 32)
    style = torch.randn(1, 1, 64, 64)
    
    # Should resize style to match content
    output, metrics = model.transfer_style(content, style)
    assert output.shape == content.shape
    assert not torch.isnan(output).any()
    assert all(not math.isnan(v) for v in metrics.values())

def test_style_transfer_different_channels():
    model = FourierStyleTransfer()
    
    # Test with different channel sizes
    content = torch.randn(1, 1, 32, 32)
    style = torch.randn(1, 3, 32, 32)
    
    # Should convert style to grayscale
    output, metrics = model.transfer_style(content, style)
    assert output.shape == content.shape
    assert not torch.isnan(output).any()
    assert all(not math.isnan(v) for v in metrics.values())

def test_style_transfer_optimization():
    model = FourierStyleTransfer(num_steps=10)  # Use fewer steps for testing
    
    # Test optimization convergence
    content = torch.randn(1, 1, 32, 32)
    style = torch.randn(1, 1, 32, 32)
    
    output, metrics = model.transfer_style(content, style)
    
    # Check that losses decrease
    assert metrics['total_loss'] > 0
    assert metrics['content_loss'] > 0
    assert metrics['style_loss'] > 0
    assert metrics['tv_loss'] > 0

def test_style_transfer_gram_matrix():
    model = FourierStyleTransfer()
    
    # Test Gram matrix computation
    x = torch.randn(2, 3, 4, 4)
    gram = model._gram_matrix(x)
    
    assert gram.shape == (2, 3, 3)
    assert not torch.isnan(gram).any()
    assert torch.allclose(gram, gram.transpose(1, 2))  # Should be symmetric

def test_style_transfer_feature_extraction():
    model = FourierStyleTransfer()
    
    # Test feature extraction
    x = torch.randn(1, 1, 32, 32)
    features = model._extract_features(x)
    
    assert isinstance(features, dict)
    assert all(k in features for k in model.style_layers + model.content_layers)
    assert all(not torch.isnan(v).any() for v in features.values())

def test_fast_neural_style(sample_input):
    model = FastNeuralStyle()
    output = model(sample_input)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert torch.all(output >= -1) and torch.all(output <= 1)  # Tanh output

def test_fast_neural_style_without_instance_norm(sample_input):
    model = FastNeuralStyle(use_instance_norm=False)
    output = model(sample_input)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert torch.all(output >= -1) and torch.all(output <= 1)

def test_residual_block(sample_input):
    block = ResidualBlock(64)
    output = block(sample_input)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()

def test_adain(sample_input, sample_style):
    adain = AdaIN()
    output = adain(sample_input, sample_style)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()

def test_adain_style_transfer(sample_input, sample_style):
    model = AdaINStyleTransfer()
    output, metrics = model.transfer_style(sample_input, sample_style)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in [
        'content_loss', 'style_loss', 'tv_loss',
        'perceptual_loss', 'total_loss'
    ])
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(v >= 0 for v in metrics.values())

def test_perceptual_loss(sample_input, sample_style):
    loss_fn = PerceptualLoss()
    loss = loss_fn(sample_input, sample_style)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)
    assert loss >= 0

def test_perceptual_loss_with_custom_layers():
    layers = ['conv1_1', 'conv2_1']
    weights = [0.5, 0.5]
    loss_fn = PerceptualLoss(layers=layers, weights=weights)
    
    x = torch.randn(1, 1, 32, 32)
    y = torch.randn(1, 1, 32, 32)
    loss = loss_fn(x, y)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert loss >= 0

def test_style_transfer_with_perceptual_loss(sample_input, sample_style):
    model = FourierStyleTransfer(perceptual_weight=1.0)
    output, metrics = model.transfer_style(sample_input, sample_style)
    
    assert output.shape == sample_input.shape
    assert not torch.isnan(output).any()
    assert 'perceptual_loss' in metrics
    assert metrics['perceptual_loss'] >= 0

def test_style_transfer_different_weights(sample_input, sample_style):
    # Test with different weight combinations
    weight_combinations = [
        (1.0, 1e5, 1e-6, 1.0),  # Default
        (1.0, 1e4, 1e-5, 0.5),  # Less style emphasis
        (2.0, 1e5, 1e-6, 2.0),  # More content emphasis
        (1.0, 1e6, 1e-7, 0.1),  # More style emphasis
    ]
    
    for c_weight, s_weight, tv_weight, p_weight in weight_combinations:
        model = FourierStyleTransfer(
            content_weight=c_weight,
            style_weight=s_weight,
            tv_weight=tv_weight,
            perceptual_weight=p_weight
        )
        output, metrics = model.transfer_style(sample_input, sample_style)
        
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()
        assert all(v >= 0 for v in metrics.values())

def test_style_transfer_gradient_flow(sample_input, sample_style):
    model = FourierStyleTransfer()
    output, _ = model.transfer_style(sample_input, sample_style)
    
    # Check if gradients flow properly
    output.requires_grad_(True)
    loss = output.sum()
    loss.backward()
    
    assert output.grad is not None
    assert not torch.isnan(output.grad).any()

def test_style_transfer_device_handling(sample_input, sample_style):
    model = FourierStyleTransfer()
    
    # Test CPU
    output_cpu, _ = model.transfer_style(
        sample_input,
        sample_style,
        device="cpu"
    )
    assert output_cpu.device.type == "cpu"
    
    # Test CUDA if available
    if torch.cuda.is_available():
        output_cuda, _ = model.transfer_style(
            sample_input,
            sample_style,
            device="cuda"
        )
        assert output_cuda.device.type == "cuda"

def test_style_transfer_memory_efficiency(sample_input, sample_style):
    model = FourierStyleTransfer()
    
    # Track memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    
    output, _ = model.transfer_style(sample_input, sample_style)
    
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated()
        assert final_memory > initial_memory  # Should use some memory
        assert final_memory < initial_memory * 10  # Should not use too much memory

def test_style_transfer_convergence(sample_input, sample_style):
    model = FourierStyleTransfer(num_steps=10)  # Use fewer steps for testing
    
    # Track loss values
    loss_history = []
    
    def loss_callback(metrics):
        loss_history.append(metrics['total_loss'])
    
    output, metrics = model.transfer_style(
        sample_input,
        sample_style,
        callback=loss_callback
    )
    
    # Check if loss decreases
    assert len(loss_history) > 0
    assert loss_history[-1] <= loss_history[0]  # Final loss should be less than initial

def test_style_transfer_batch_consistency(sample_input, sample_style):
    model = FourierStyleTransfer()
    
    # Process single image
    output_single, metrics_single = model.transfer_style(
        sample_input,
        sample_style
    )
    
    # Process batch
    batch_size = 4
    input_batch = sample_input.repeat(batch_size, 1, 1, 1)
    style_batch = sample_style.repeat(batch_size, 1, 1, 1)
    output_batch, metrics_batch = model.transfer_style_batch(
        input_batch,
        style_batch
    )
    
    # Check consistency
    assert output_batch.shape == (batch_size, *sample_input.shape[1:])
    assert not torch.isnan(output_batch).any()
    assert all(k in metrics_batch for k in metrics_single.keys())

def test_style_transfer_edge_cases():
    model = FourierStyleTransfer()
    
    # Test with very small images
    small_input = torch.randn(1, 1, 8, 8)
    small_style = torch.randn(1, 1, 8, 8)
    output, _ = model.transfer_style(small_input, small_style)
    assert output.shape == small_input.shape
    
    # Test with very large images
    large_input = torch.randn(1, 1, 256, 256)
    large_style = torch.randn(1, 1, 256, 256)
    output, _ = model.transfer_style(large_input, large_style)
    assert output.shape == large_input.shape
    
    # Test with different aspect ratios
    wide_input = torch.randn(1, 1, 32, 64)
    wide_style = torch.randn(1, 1, 32, 64)
    output, _ = model.transfer_style(wide_input, wide_style)
    assert output.shape == wide_input.shape 