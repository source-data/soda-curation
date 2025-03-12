from abc import ABC, abstractmethod


class View(ABC):
    @abstractmethod
    def render(self):
        pass
