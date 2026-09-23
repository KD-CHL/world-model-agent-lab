"""Load a project checkpoint from its declared serialization format."""
from pathlib import Path


def load_dynamics(path, *, device='cpu'):
    checkpoint = Path(path)
    if checkpoint.suffix.lower() in ('.pt', '.pth'):
        from wmal.models.neural_dynamics import NeuralJointDynamics
        return NeuralJointDynamics.load(checkpoint, device=device)
    from wmal.models.latent_dynamics import JointDynamics
    return JointDynamics.load(checkpoint)
