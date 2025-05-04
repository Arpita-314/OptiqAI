import torch
import os

class ModelDeployment:
    def __init__(self, model: torch.nn.Module, export_dir: str = "exports"):
        """
        Initialize the deployment module.

        Args:
        model (torch.nn.Module): Trained model instance
        export_dir (str): Directory to save exported models
        """
        self.model = model
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def export_to_torchscript(self, example_input: torch.Tensor) -> str:
        """
        Export the model to TorchScript format.

        Args:
        example_input (torch.Tensor): Example input tensor for tracing

        Returns:
        str: Path to the exported TorchScript model
        """
        print("Exporting model to TorchScript...")
        scripted_model = torch.jit.trace(self.model, example_input)
        torchscript_path = os.path.join(self.export_dir, "model_torchscript.pt")
        scripted_model.save(torchscript_path)
        print(f"Model saved to {torchscript_path}")
        return torchscript_path

    def export_to_onnx(self, example_input: torch.Tensor) -> str:
        """
        Export the model to ONNX format.

        Args:
        example_input (torch.Tensor): Example input tensor for tracing

        Returns:
        str: Path to the exported ONNX model
        """
        print("Exporting model to ONNX...")
        onnx_path = os.path.join(self.export_dir, "model.onnx")
        
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # Export the model
        torch.onnx.export(
            self.model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        
        print(f"Model saved to {onnx_path}")
        return onnx_path

    def deploy_summary(self):
        """Print a summary of available deployment options."""
        print("\n[Deployment Options]")
        print("1. TorchScript: Optimized for PyTorch runtime and edge devices.")
        print("2. ONNX: Cross-platform compatibility with frameworks like TensorRT and ONNX Runtime.")

# Example usage
if __name__ == "__main__":
    # Simulated trained model (replace with your actual trained model)
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x * 2
    
    dummy_model = DummyModel()
    example_input = torch.randn(1, 1, 256, 256)

    deployment = ModelDeployment(dummy_model)
    deployment.deploy_summary()
    
    # Export to TorchScript and ONNX
    deployment.export_to_torchscript(example_input)
    deployment.export_to_onnx(example_input)
