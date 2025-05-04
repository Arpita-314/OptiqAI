import torch
import numpy as np
import os
import time
from fourierlab.physics.quantum_optics import (
    QuantumOpticsCalculator,
    QuantumStateGenerator,
    QuantumOperations,
    QuantumStateTomography,
    QuantumStateAnalyzer,
    QuantumStateIO
)

def test_edge_cases():
    """Test edge cases for quantum states and operations"""
    calculator = QuantumOpticsCalculator()
    generator = QuantumStateGenerator(calculator)
    operations = QuantumOperations(calculator)
    
    # Test Fock state with n=0 (vacuum state)
    vacuum = generator.create_fock_state(n=0)
    assert torch.allclose(torch.abs(vacuum.state), torch.exp(-vacuum.state.real**2/2))
    
    # Test Cat state with alpha=0 (coherent state)
    coherent = generator.create_cat_state(alpha=0.0)
    assert torch.allclose(torch.abs(coherent.state), torch.exp(-coherent.state.real**2/2))
    
    # Test NOON state with n=1 (Bell state)
    bell = generator.create_noon_state(n=1)
    assert torch.allclose(torch.sum(torch.abs(bell.state)**2), torch.tensor(1.0))
    
    # Test operations with zero parameters
    state = vacuum.state
    assert torch.allclose(operations.phase_shifter(state, phi=0.0), state)
    assert torch.allclose(operations.squeezer(state, r=0.0), state)
    assert torch.allclose(operations.displacement(state, alpha=0.0), state)
    assert torch.allclose(operations.kerr_nonlinearity(state, chi=0.0), state)
    
    # Test operations with extreme parameters
    assert torch.allclose(operations.phase_shifter(state, phi=2*np.pi), state)
    assert torch.allclose(operations.beam_splitter(state, state, theta=0.0), (state, state))
    assert torch.allclose(operations.beam_splitter(state, state, theta=np.pi/2), (state, state))

def test_performance():
    """Test performance with large-scale simulations"""
    calculator = QuantumOpticsCalculator()
    generator = QuantumStateGenerator(calculator)
    operations = QuantumOperations(calculator)
    
    # Test with large state size
    large_size = (512, 512)
    large_fock = generator.create_fock_state(n=2, size=large_size)
    large_cat = generator.create_cat_state(alpha=2.0, size=large_size)
    
    # Test operations on large states
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    out1, out2 = operations.beam_splitter(large_fock.state, large_cat.state)
    end_time.record()
    torch.cuda.synchronize()
    print(f"Beam splitter operation time: {start_time.elapsed_time(end_time):.2f} ms")
    
    # Test tomography with large number of measurements
    tomography = QuantumStateTomography(calculator)
    measurements = tomography.measure_quadratures(large_fock.state, num_measurements=10000)
    reconstructed_state = tomography.reconstruct_state(measurements)
    assert tomography.calculate_fidelity(large_fock.state, reconstructed_state) > 0.9

def test_performance_benchmarks():
    """Test performance of quantum optics operations"""
    calculator = QuantumOpticsCalculator()
    generator = QuantumStateGenerator(calculator)
    operations = QuantumOperations(calculator)
    
    # Test large state generation
    print("\nTesting large state generation...")
    start_time = time.time()
    large_fock = generator.generate_fock_state(n=100, size=1000)
    print(f"Large Fock state generation: {time.time() - start_time:.3f} seconds")
    
    # Test beam splitter performance
    print("\nTesting beam splitter performance...")
    start_time = time.time()
    for _ in range(100):
        out1, out2 = operations.beam_splitter(large_fock, large_fock)
    print(f"100 beam splitter operations: {time.time() - start_time:.3f} seconds")
    
    # Test phase shifter performance
    print("\nTesting phase shifter performance...")
    start_time = time.time()
    for _ in range(100):
        out = operations.phase_shifter(large_fock, np.pi/4)
    print(f"100 phase shifter operations: {time.time() - start_time:.3f} seconds")
    
    # Test squeezing performance
    print("\nTesting squeezing performance...")
    start_time = time.time()
    for _ in range(100):
        out = operations.squeezer(large_fock, 0.5)
    print(f"100 squeezing operations: {time.time() - start_time:.3f} seconds")
    
    # Test displacement performance
    print("\nTesting displacement performance...")
    start_time = time.time()
    for _ in range(100):
        out = operations.displacement(large_fock, 1+1j)
    print(f"100 displacement operations: {time.time() - start_time:.3f} seconds")
    
    # Test Kerr nonlinearity performance
    print("\nTesting Kerr nonlinearity performance...")
    start_time = time.time()
    for _ in range(100):
        out = operations.kerr_nonlinearity(large_fock, 0.1)
    print(f"100 Kerr nonlinearity operations: {time.time() - start_time:.3f} seconds")
    
    # Test two-mode squeezing performance
    print("\nTesting two-mode squeezing performance...")
    start_time = time.time()
    for _ in range(100):
        out1, out2 = operations.two_mode_squeezing(large_fock, large_fock, 0.5)
    print(f"100 two-mode squeezing operations: {time.time() - start_time:.3f} seconds")
    
    # Test state analysis performance
    print("\nTesting state analysis performance...")
    start_time = time.time()
    for _ in range(100):
        var_x, var_p = calculator.calculate_quadrature_variances(large_fock)
        n = calculator.calculate_photon_number(large_fock)
        r = calculator.calculate_squeezing(large_fock)
        E = calculator.calculate_entanglement(large_fock)
        F = calculator.calculate_quantum_fisher_information(large_fock)
        G = calculator.calculate_metrological_gain(large_fock)
    print(f"100 state analysis operations: {time.time() - start_time:.3f} seconds")

def test_quantum_optics_features():
    # Initialize calculator
    calculator = QuantumOpticsCalculator()
    generator = QuantumStateGenerator(calculator)
    operations = QuantumOperations(calculator)
    analyzer = QuantumStateAnalyzer(calculator)
    io = QuantumStateIO(calculator)
    
    # Create output directory for visualizations and data
    output_dir = "quantum_optics_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Test state generation and properties
    print("Testing state generation...")
    
    # Test Fock state
    fock_state = generator.create_fock_state(n=2)
    assert fock_state.state.dtype == torch.complex64
    assert torch.allclose(torch.sum(torch.abs(fock_state.state)**2), torch.tensor(1.0))
    
    # Test Cat state
    cat_state = generator.create_cat_state(alpha=2.0)
    assert cat_state.state.dtype == torch.complex64
    assert torch.allclose(torch.sum(torch.abs(cat_state.state)**2), torch.tensor(1.0))
    
    # Test GKP state
    gkp_state = generator.create_gkp_state(delta=0.1)
    assert gkp_state.state.dtype == torch.complex64
    assert torch.allclose(torch.sum(torch.abs(gkp_state.state)**2), torch.tensor(1.0))
    
    # Test NOON state
    noon_state = generator.create_noon_state(n=3)
    assert noon_state.state.dtype == torch.complex64
    assert torch.allclose(torch.sum(torch.abs(noon_state.state)**2), torch.tensor(1.0))
    
    # Test Cluster state
    cluster_state = generator.create_cluster_state(num_modes=4)
    assert cluster_state.state.dtype == torch.complex64
    assert torch.allclose(torch.sum(torch.abs(cluster_state.state)**2), torch.tensor(1.0))
    
    # 2. Test quantum operations
    print("\nTesting quantum operations...")
    
    # Test beam splitter
    state1 = fock_state.state
    state2 = cat_state.state
    out1, out2 = operations.beam_splitter(state1, state2, theta=np.pi/4)
    assert out1.dtype == torch.complex64
    assert out2.dtype == torch.complex64
    
    # Test phase shift
    shifted_state = operations.phase_shifter(state1, phi=np.pi/2)
    assert shifted_state.dtype == torch.complex64
    
    # Test squeezing
    squeezed_state = operations.squeezer(state1, r=0.5)
    assert squeezed_state.dtype == torch.complex64
    
    # Test displacement
    displaced_state = operations.displacement(state1, alpha=1.0+1.0j)
    assert displaced_state.dtype == torch.complex64
    
    # Test Kerr nonlinearity
    kerr_state = operations.kerr_nonlinearity(state1, chi=0.1)
    assert kerr_state.dtype == torch.complex64
    
    # Test two-mode squeezing
    squeezed1, squeezed2 = operations.two_mode_squeezing(state1, state2, r=0.5)
    assert squeezed1.dtype == torch.complex64
    assert squeezed2.dtype == torch.complex64
    
    # 3. Test tomography
    print("\nTesting state tomography...")
    tomography = QuantumStateTomography(calculator)
    measurements = tomography.measure_quadratures(state1, num_measurements=1000)
    reconstructed_state = tomography.reconstruct_state(measurements)
    fidelity = tomography.calculate_fidelity(state1, reconstructed_state)
    print(f"State reconstruction fidelity: {fidelity:.4f}")
    assert fidelity > 0.9  # High fidelity expected for good reconstruction
    
    # 4. Test analysis and visualization
    print("\nTesting state analysis and visualization...")
    
    # Analyze states
    fock_metrics = analyzer.analyze_state(fock_state)
    cat_metrics = analyzer.analyze_state(cat_state)
    gkp_metrics = analyzer.analyze_state(gkp_state)
    noon_metrics = analyzer.analyze_state(noon_state)
    cluster_metrics = analyzer.analyze_state(cluster_state)
    
    # Verify metrics
    for metrics in [fock_metrics, cat_metrics, gkp_metrics, noon_metrics, cluster_metrics]:
        assert 'wigner' in metrics
        assert 'phase_space' in metrics
        assert 'entropy' in metrics
        assert 'photon_number' in metrics
        assert 'squeezing' in metrics
        assert 'entanglement' in metrics
        assert 'fisher_info' in metrics
        assert 'metrological_gain' in metrics
    
    # Visualize results
    analyzer.visualize_analysis(fock_metrics, save_dir=os.path.join(output_dir, "fock_state"))
    analyzer.visualize_analysis(cat_metrics, save_dir=os.path.join(output_dir, "cat_state"))
    analyzer.visualize_analysis(gkp_metrics, save_dir=os.path.join(output_dir, "gkp_state"))
    analyzer.visualize_analysis(noon_metrics, save_dir=os.path.join(output_dir, "noon_state"))
    analyzer.visualize_analysis(cluster_metrics, save_dir=os.path.join(output_dir, "cluster_state"))
    
    # 5. Test IO functionality
    print("\nTesting IO functionality...")
    
    # Test state export/import
    state_file = os.path.join(output_dir, "fock_state.npy")
    io.export_state(fock_state.state, state_file, format='numpy')
    imported_state = io.import_state(state_file, format='numpy')
    assert torch.allclose(fock_state.state, imported_state)
    
    # Test measurements export/import
    measurements_file = os.path.join(output_dir, "measurements.npz")
    io.export_measurements(measurements, measurements_file, format='numpy')
    imported_measurements = io.import_measurements(measurements_file, format='numpy')
    assert all(torch.allclose(measurements[k], imported_measurements[k]) for k in measurements.keys())
    
    print("\nAll tests completed successfully! Results saved in:", output_dir)

if __name__ == "__main__":
    test_edge_cases()
    test_performance()
    test_performance_benchmarks()
    test_quantum_optics_features() 