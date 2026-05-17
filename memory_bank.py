import torch
from collections import deque
from logzero import logger

class MemoryBank:
    """
    A memory bank that stores a deque of feature tensors on a specified device.
    It automatically manages its size by discarding the oldest tensors when full.
    """

    def __init__(self, max_size: int = 256, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.deque = deque()
        logger.info(f"Initialized MemoryBank with max_size={self.max_size} on device='{self.device}'")

    def clear(self):
        """Clears all tensors from the memory bank."""
        self.deque.clear()
        logger.debug("MemoryBank cleared.")

    def add(self, features: torch.Tensor):
        """
        Adds new feature tensors to the bank.
        If the bank exceeds its max_size, it removes the oldest tensors.
        """
        # Add new features one by one to the right of the deque
        for i in range(features.shape[0]):
            self.deque.append(features[i].detach()) # Use detach to prevent holding onto graph

        # Evict old features from the left if oversized
        while len(self.deque) > self.max_size:
            self.deque.popleft()

    def get_all(self) -> torch.Tensor:
        """
        Returns all features currently in the memory bank as a single stacked tensor.
        """
        if not self.deque:
            # Return an empty tensor with a placeholder for feature dimension if the bank is empty
            return torch.empty((0, 1024), device=self.device) # Assuming a feature dim, adjust if needed
        
        # Stack the deque of tensors into a single tensor
        return torch.stack(list(self.deque), dim=0).to(self.device)

    def __len__(self):
        return len(self.deque)
