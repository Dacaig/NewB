import syft
import collections
import torch
import numpy as np
from poly import *
import random
class fl_client(syft.VirtualWorker):
    super
    def get_parameters(self, config) -> collections.List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: collections.List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

class leader_client(fl_client):
    def __init__(self, max_int=10):
        self.max_int = max_int

    def generate_random_int(self):
        return random.randint(0, self.max_int - 1)

    def add_random_ints_to_vector(self, vector):
        if not isinstance(vector, np.ndarray):
            raise TypeError("The input vector must be a numpy ndarray.")
        random_ints = np.array([self.generate_random_int() for _ in range(len(vector))])
        return vector + random_ints
