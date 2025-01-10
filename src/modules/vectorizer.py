from abc import ABC, abstractmethod
import compress_fasttext
import numpy as np

class Vec(ABC):
    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def most_similar(self, text:str, k:int=10)->list[tuple[str, float]]:
        raise NotImplementedError
    
    @abstractmethod
    def get_vector(self, text:str)->np.ndarray:
        raise NotImplementedError


class CustomFasttext(Vec):
    def __init__(self, model_path_or_link:str):
        self.model = None
        self.model_path_or_link = model_path_or_link
        self.load()
                
    def load(self):
        self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(self.model_path_or_link)

    def most_similar(self, text:str, k:int=10)->list[tuple[str, float]]:
        return self.model.most_similar(positive=[text], topn=k)
    
    def get_vector(self, text:str)->np.ndarray:
        return self.model.get_vector(text)