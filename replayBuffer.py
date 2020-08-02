import random

class replayBuffer():
    """ This class describes the replay buffer """

    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        """ Rotating list """
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        
        return random.sample(self.buffer, batch_size)
        # indices = random.sample(range(self.size), batch_size)
        # return [self.buffer[index] for index in indices]

