import random

class replayBuffer():
    """ This class describes a list of experiences"""
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        """Rotating list. Evict oldest items if list if full"""
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        """Return random sample of our experiences of size batch_size 
        If we don't have at least batch_size worth of experiences, 
        return as many as possible"""

        if (self.size <= self.max_size and self.index > batch_size):
            return random.sample(self.buffer[:self.index - 1], batch_size)
        elif (self.index <= batch_size):
            return self.buffer[:self.index - 1]

        return random.sample(self.buffer, batch_size)

