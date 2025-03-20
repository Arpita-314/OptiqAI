import torch
from data.ingestion import DataIngestion
from preprocessing.fourier_preprocessing import FourierPreprocessing
from models.architecture import ModelSelector
from training.trainer import OpticsTrainer
from utils.metrics import FourierOpticsMetrics
from utils.visualization import FourierOpticsVisualization
from utils.deployment import ModelDeployment

def test_pipeline():
    print("Starting end-to-end testing of the Fourier Optics AutoML Framework...")

    # Step 1: Data Ingestion
    print("\n[Step 1: Data Ingestion]")
    ingestion = DataIngestion()
    data_type = "complex_field"  # Simulated user input
    pixel_size = 5.0  # μm
    wavelength = 632.8  # nm

    # Simulate a complex wavefront dataset (256x256)
    data = torch.randn(256, 256) + 1j * torch.randn(256, 256)
    print(f"Data type: {data_type}, Pixel size: {pixel_size} μm, Wavelength: {wavelength} nm")

    # Step 2: Preprocessing
    print("\n[Step 2: Preprocessing]")
    preprocessor = FourierPreprocessing(pixel_size=pixel_size, wavelength=wavelength)
    amplitude, phase = preprocessor.preprocess(data, remove_dc=True, window_type='tukey', unwrap_phase=True)
    print(f"Amplitude shape: {amplitude.shape}, Phase shape: {phase.shape}")

    # Step 3: Model Selection
    print("\n[Step 3: Model Selection]")
    selector = ModelSelector(data_shape=amplitude.shape, data_type=data_type)
    recommended_model = selector.auto_recommend()
    print(f"Recommended model: {selector.available_models[recommended_model]}")
    model = selector.build_model()

    # Step 4: Training
    print("\n[Step 4: Training]")
    trainer = OpticsTrainer(model=model, wavelength=wavelength * 1e-9, pixel_size=pixel_size * 1e-6)

    # Simulate training and validation datasets (100 training samples, 20 validation samples)
    train_data = torch.randn(100, 1, 256, 256)  # Batch size of 100 (amplitude + phase)
    train_targets = torch.randn(100, 2, 256, 256)  # Amplitude and phase as targets
    val_data = torch.randn(20, 1, 256, 256)
    val_targets = torch.randn(20, 2, 256, 256)

    train_loader = trainer.create_dataloader(train_data, train_targets)
    val_loader = trainer.create_dataloader(val_data, val_targets)

    config = {
        "epochs": 5,
        "patience": 3,
        "batch_size": 8,
        "auto_tune": False,
    }

    print("Starting manual training...")
    trainer.manual_train(train_loader, val_loader, config)

    # Step 5: Validation and Metrics Calculation
    print("\n[Step 5: Validation and Metrics Calculation]")
    metrics_calculator = FourierOpticsMetrics(wavelength=wavelength * 1e-9, pixel_size=pixel_size * 1e-6)

    # Simulate predicted and target wavefronts for validation metrics
    predicted_wavefront = torch.exp(1j * torch.randn(256, 256))
    target_wavefront = torch.exp(1j * (torch.randn(256, 256) * 0.9 + 0.1))

    results = metrics_calculator.calculate_all_metrics(predicted_wavefront, target_wavefront)
    
    print("Validation Metrics:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    # Step 6: Visualization
    print("\n[Step 6: Visualization]")
    
    viz = FourierOpticsVisualization()
    
    # Visualize wavefront comparison
    viz.compare_wavefronts(predicted_wavefront, target_wavefront)

    # Simulate MTF data for visualization purposes
    mtf_example = np.random.rand(256, 256) 
    viz.plot_mtf(mtf_example)

# Deployment step 
deployment = ModelDeployment(model)
deployment.export_to_onnx("model.onnx")
