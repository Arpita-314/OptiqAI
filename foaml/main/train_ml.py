from physics.components import Grating
from models.surrogate import FieldPredictor  # Your existing ML model
from preprocessing.field_processing import normalize_field

# Generate training data using physics
grating = Grating(period=1e-6)
x = np.linspace(0, 10e-6, 1000)
fields = [grating.compute_field(x, wl) for wl in np.linspace(400e-9, 800e-9, 100)]
fields = normalize_field(fields)  # New preprocessing step

# Train ML model (your existing code)
model = FieldPredictor()
model.train(fields)