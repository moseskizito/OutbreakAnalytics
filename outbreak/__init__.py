"""outbreak: early-warning analytics for community and facility surveillance data."""
from .simulate import simulate_surveillance
from .detect import ALGORITHMS
from .evaluate import evaluate_algorithm, lead_time

__all__ = ["simulate_surveillance", "ALGORITHMS", "evaluate_algorithm", "lead_time"]
__version__ = "0.1.0"
