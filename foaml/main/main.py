import torch
from data.ingestion import DataIngestion
from preprocessing.fourier_preprocessing import FourierPreprocessing
from models.architecture import ModelSelector
from training.trainer import OpticsTrainer
from utils.deployment import ModelDeployment
from utils.visualization import FourierOpticsVisualization

def main():
    print("Welcome to Fourier Optics AutoML Framework")
    
    # Initialize data ingestion
    ingestion = DataIngestion()
    
    # Get user input
    data_type, pixel_size, wavelength = ingestion.get_user_input()
    
    # Here you would typically load your data
    # data = ingestion.load_data('path/to/your/data.mat')
    
    # For demonstration, let's create some dummy data
    data = torch.randn(256, 256) + 1j * torch.randn(256, 256)
    
    print("Data ingestion complete.")
    
    # Initialize preprocessing
    preprocessor = FourierPreprocessing(pixel_size=pixel_size, wavelength=wavelength)
    
    # Preprocess the data
    if data_type == 'complex_field':
        amplitude, phase = preprocessor.preprocess(data, remove_dc=True, window_type='tukey', unwrap_phase=True)
        print(f"Preprocessed amplitude shape: {amplitude.shape}")
        print(f"Preprocessed phase shape: {phase.shape}")
    else:
        preprocessed_data = preprocessor.preprocess(data, remove_dc=True, window_type='tukey')
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
    
    print("Preprocessing complete.")
    
    # Model Selection
    selector = ModelSelector(data_shape=amplitude.shape, data_type=data_type)
    
    # Auto-recommend first
    recommended_model = selector.auto_recommend()
    print(f"\nRecommended model: {selector.available_models[recommended_model]}")
    
    # User interaction
    use_recommended = input("Use recommended model? (y/n): ").lower()
    if use_recommended == 'y':
        selector.model_choice = recommended_model
    else:
        selector.get_user_preferences()
    
    # Build model
    model = selector.build_model()
    
    print(f"\nSelected model architecture:\n{model}")
    print("Next steps: Training and optimization")

    # Training setup
    trainer = OpticsTrainer(model, wavelength, pixel_size)
    
    # Create sample data (replace this with your actual data)
    train_data = torch.randn(100, 1, 256, 256)  # Batch, channels, height, width
    train_targets = torch.randn(100, 2, 256, 256)  # Amplitude + phase
    val_data = torch.randn(20, 1, 256, 256)
    val_targets = torch.randn(20, 2, 256, 256)
    
    train_loader = trainer.create_dataloader(train_data, train_targets, batch_size=8)
    val_loader = trainer.create_dataloader(val_data, val_targets, batch_size=8, shuffle=False)
    
    # Get user settings
    config = trainer.get_user_settings()
    
    if config["auto_tune"]:
        print("Running hyperparameter optimization...")
        trainer.auto_tune(train_loader, val_loader)
    else:
        print("Starting manual training...")
        trainer.manual_train(train_loader, val_loader, config)
    
    print("Training complete. Best model weights saved.")
    
    # Deployment setup
    print("Next steps: Validation and deployment")

deployment = ModelDeployment(model)
deployment.deploy_summary()
    
# Example input for tracing during export (replace with actual input shape)
example_input = torch.randn(1, 1, 256, 256)  # Batch size of 1
    
# Get user choice for deployment format
deploy_choice = input("Choose deployment format (torchscript/onnx/both): ").lower()
    
if deploy_choice == "torchscript" or deploy_choice == "both":
    deployment.export_to_torchscript(example_input)
    
if deploy_choice == "onnx" or deploy_choice == "both":
    deployment.export_to_onnx(example_input)
    
print("\nDeployment complete. Models exported successfully.")

print("Training complete. Best model weights saved.")
    
    # Validation and visualization setup
    print("\nNext steps: Visualization")
    
    viz = FourierOpticsVisualization()
    
    # Simulate validation outputs for visualization purposes
    predicted_wavefront = torch.exp(1j * torch.randn(256, 256))  # Replace with actual model predictions
    target_wavefront = torch.exp(1j * torch.randn(256, 256))     # Replace with validation targets
    
    # Visualize results
    viz.compare_wavefronts(predicted_wavefront, target_wavefront)

    # Simulate MTF data for visualization purposes
    mtf_example = np.random.rand(256, 256)  # Replace with actual MTF data from metrics module
    viz.plot_mtf(mtf_example)
    
if __name__ == "__main__":
    main()
