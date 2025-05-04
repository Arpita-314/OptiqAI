import torch
import numpy as np
import pytest
from fourierlab.physics.quantum_optics import (
    QuantumOpticsCalculator,
    QuantumStateGenerator,
    QuantumStateAnalyzer,
    QuantumOperations,
    FockState,
    CatState,
    GKPState,
    NOONState,
    ClusterState
)

@pytest.fixture
def calculator():
    return QuantumOpticsCalculator()

@pytest.fixture
def generator(calculator):
    return QuantumStateGenerator(calculator)

@pytest.fixture
def analyzer(calculator):
    return QuantumStateAnalyzer(calculator)

def test_fock_state_creation(generator):
    """Test Fock state creation"""
    n = 2
    state = generator.create_fock_state(n)
    
    assert isinstance(state, FockState)
    assert state.n == n
    assert state.state is not None
    assert state.state.shape == (256, 256)

def test_cat_state_creation(generator):
    """Test cat state creation"""
    alpha = 2.0
    state = generator.create_cat_state(alpha)
    
    assert isinstance(state, CatState)
    assert state.alpha == alpha
    assert state.state is not None
    assert state.state.shape == (256, 256)

def test_gkp_state_creation(generator):
    """Test GKP state creation"""
    delta = 0.2
    state = generator.create_gkp_state(delta)
    
    assert isinstance(state, GKPState)
    assert state.delta == delta
    assert state.state is not None
    assert state.state.shape == (256, 256)

def test_noon_state_creation(generator):
    """Test NOON state creation"""
    n = 2
    state = generator.create_noon_state(n)
    
    assert isinstance(state, NOONState)
    assert state.n == n
    assert state.state is not None
    assert state.state.shape == (256, 256)

def test_cluster_state_creation(generator):
    """Test cluster state creation"""
    num_modes = 4
    state = generator.create_cluster_state(num_modes=num_modes)
    
    assert isinstance(state, ClusterState)
    assert state.num_modes == num_modes
    assert state.state is not None
    assert state.state.shape == (256, 256)

def test_wigner_function(calculator):
    """Test Wigner function calculation"""
    state = FockState(n=1)
    wigner = state.calculate_wigner()
    
    assert wigner is not None
    assert wigner.shape == (256, 256)
    assert torch.all(torch.isfinite(wigner))

def test_phase_space(calculator):
    """Test phase space calculation"""
    state = CatState(alpha=2.0)
    x, p, Q = state.calculate_phase_space()
    
    assert x is not None
    assert p is not None
    assert Q is not None
    assert x.shape == (256, 256)
    assert p.shape == (256, 256)
    assert Q.shape == (256, 256)

def test_quantum_properties(calculator):
    """Test quantum property calculations"""
    state = FockState(n=2)
    state_tensor = state.state.unsqueeze(0).unsqueeze(0)
    
    # Test quadrature variances
    var_x, var_p = calculator.calculate_quadrature_variances(state_tensor)
    assert torch.all(torch.isfinite(var_x))
    assert torch.all(torch.isfinite(var_p))
    
    # Test photon number
    n = calculator.calculate_photon_number(state_tensor)
    assert torch.all(torch.isfinite(n))
    
    # Test squeezing
    r = calculator.calculate_squeezing(state_tensor)
    assert torch.all(torch.isfinite(r))

def test_entanglement(calculator):
    """Test entanglement calculation"""
    state = CatState(alpha=2.0)
    state_tensor = state.state.unsqueeze(0).unsqueeze(0)
    
    entanglement = calculator.calculate_entanglement(state_tensor)
    assert torch.all(torch.isfinite(entanglement))

def test_fisher_information(calculator):
    """Test quantum Fisher information calculation"""
    state = GKPState(delta=0.2)
    state_tensor = state.state.unsqueeze(0).unsqueeze(0)
    
    # Test different parameters
    for param in ['phase', 'amplitude', 'squeezing']:
        F = calculator.calculate_quantum_fisher_information(state_tensor, param)
        assert torch.all(torch.isfinite(F))

def test_metrological_gain(calculator):
    """Test metrological gain calculation"""
    state = FockState(n=1)
    state_tensor = state.state.unsqueeze(0).unsqueeze(0)
    
    gain = calculator.calculate_metrological_gain(state_tensor)
    assert torch.all(torch.isfinite(gain))

def test_state_analysis(analyzer):
    """Test complete state analysis"""
    state = FockState(n=2)
    metrics = analyzer.analyze_state(state)
    
    # Check all metrics are present
    required_metrics = [
        'wigner', 'phase_space', 'entropy',
        'photon_number', 'squeezing', 'entanglement',
        'fisher_info', 'metrological_gain'
    ]
    
    for metric in required_metrics:
        assert metric in metrics
        assert torch.all(torch.isfinite(metrics[metric]))

def test_state_normalization():
    """Test state normalization"""
    # Test Fock state
    fock = FockState(n=2)
    assert torch.allclose(torch.sum(fock.state**2), torch.tensor(1.0), atol=1e-6)
    
    # Test cat state
    cat = CatState(alpha=2.0)
    assert torch.allclose(torch.sum(cat.state**2), torch.tensor(1.0), atol=1e-6)
    
    # Test GKP state
    gkp = GKPState(delta=0.2)
    assert torch.allclose(torch.sum(gkp.state**2), torch.tensor(1.0), atol=1e-6)

def test_state_parameters():
    """Test state parameter handling"""
    # Test invalid Fock state
    with pytest.raises(ValueError):
        FockState(n=-1)
    
    # Test invalid cat state
    with pytest.raises(ValueError):
        CatState(alpha=-1.0)
    
    # Test invalid GKP state
    with pytest.raises(ValueError):
        GKPState(delta=-0.1)

def test_calculator_parameters():
    """Test calculator parameter handling"""
    # Test invalid wavelength
    with pytest.raises(ValueError):
        QuantumOpticsCalculator(wavelength=-1.0)
    
    # Test invalid pixel size
    with pytest.raises(ValueError):
        QuantumOpticsCalculator(pixel_size=-1.0)
    
    # Test invalid Fisher information parameter
    calculator = QuantumOpticsCalculator()
    state = FockState(n=1)
    state_tensor = state.state.unsqueeze(0).unsqueeze(0)
    
    with pytest.raises(ValueError):
        calculator.calculate_quantum_fisher_information(state_tensor, 'invalid')

def test_quantum_operations():
    """Test quantum operations"""
    calculator = QuantumOpticsCalculator()
    operations = QuantumOperations(calculator)
    
    # Test beam splitter
    state1 = torch.randn(256, 256, 2)
    state2 = torch.randn(256, 256, 2)
    out1, out2 = operations.beam_splitter(state1, state2)
    assert out1.shape == state1.shape
    assert out2.shape == state2.shape
    
    # Test phase shifter
    state = torch.randn(256, 256, 2)
    phi = np.pi/4
    shifted = operations.phase_shifter(state, phi)
    assert shifted.shape == state.shape
    
    # Test squeezer
    r = 0.5
    squeezed = operations.squeezer(state, r)
    assert squeezed.shape == state.shape
    
    # Test displacement
    alpha = 1.0 + 1.0j
    displaced = operations.displacement(state, alpha)
    assert displaced.shape == state.shape

def test_state_visualization(analyzer):
    """Test state visualization"""
    state = FockState(n=2)
    visualizations = analyzer.visualize_state(state)
    
    # Check all visualizations are present
    required_visualizations = [
        'wigner', 'phase_space', 'quadratures', 'metrics'
    ]
    
    for viz in required_visualizations:
        assert viz in visualizations
        assert 'type' in visualizations[viz]
        assert 'data' in visualizations[viz]
        assert 'title' in visualizations[viz]

def test_visualization_types(analyzer):
    """Test different visualization types"""
    state = CatState(alpha=2.0)
    visualizations = analyzer.visualize_state(state)
    
    # Check visualization types
    assert visualizations['wigner']['type'] == 'heatmap'
    assert visualizations['phase_space']['type'] == 'contour'
    assert visualizations['quadratures']['type'] == 'histogram'
    assert visualizations['metrics']['type'] == 'bar'

def test_visualization_data(analyzer):
    """Test visualization data format"""
    state = GKPState(delta=0.2)
    visualizations = analyzer.visualize_state(state)
    
    # Check data formats
    assert isinstance(visualizations['wigner']['data'], np.ndarray)
    assert isinstance(visualizations['phase_space']['data'], dict)
    assert isinstance(visualizations['quadratures']['data'], dict)
    assert isinstance(visualizations['metrics']['data'], dict)

def test_operation_parameters():
    """Test operation parameter handling"""
    calculator = QuantumOpticsCalculator()
    operations = QuantumOperations(calculator)
    state = torch.randn(256, 256, 2)
    
    # Test invalid phase
    with pytest.raises(ValueError):
        operations.phase_shifter(state, np.inf)
    
    # Test invalid squeezing
    with pytest.raises(ValueError):
        operations.squeezer(state, -np.inf)
    
    # Test invalid displacement
    with pytest.raises(ValueError):
        operations.displacement(state, np.inf + 1j*np.inf)

def test_state_combinations():
    """Test combining different quantum states"""
    calculator = QuantumOpticsCalculator()
    operations = QuantumOperations(calculator)
    
    # Create states
    fock = FockState(n=1)
    cat = CatState(alpha=2.0)
    
    # Combine states through beam splitter
    out1, out2 = operations.beam_splitter(fock.state, cat.state)
    
    assert torch.all(torch.isfinite(out1))
    assert torch.all(torch.isfinite(out2))
    assert out1.shape == fock.state.shape
    assert out2.shape == cat.state.shape 