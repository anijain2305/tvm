from abc import ABC, abstractmethod
class TensorDecomposition(ABC):
    @abstractmethod
    def decompose():
        pass

    @abstractmethod
    def reconstruct():
        pass

    @abstractmethod
    def predict_flops():
        pass

    @abstractmethod
    def predict_memory():
        pass

    @abstractmethod
    def preprocess_tensor():
        pass

    @abstractmethod
    def postprocess_tensor():
        pass

    @abstractmethod
    def simulate():
        pass


