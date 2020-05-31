import abc
import io
import numpy as np


# Decay Rate = 0 => equivalent to original averaging algorithm, no alpha update
# Decay Rate = 1 => equivalent to always adding same alpha update
# In between => added alpha update gets smaller by time

# Assumed logic for go
#  If peers available, choose 1 randomly and send
#  If no peers available, buffer, do the above step for element in buffer when peers available

class GossipAggregator:
    def __init__(self, data_points, decay_rate):
        self.alpha = 0
        self.alpha_update = float(data_points) / 10000
        self.decay_rate = decay_rate

    def push_model(self, model):
        # Update alpha
        prev_alpha = self.alpha

        self.alpha += self.alpha_update
        self.alpha_update *= self.decay_rate

        print("Alpha:", prev_alpha, "->", self.alpha, "->", self.alpha / 2)

        self.alpha /= 2
        # Compress to byte array
        file = io.BytesIO()
        np.savez_compressed(file, model=model, alpha=self.alpha)
        data = file.getbuffer()
        # Send
        self._push_to_go(data)

    def receive_updates(self, model):
        # Receive
        data = self._receive_from_go()
        # Process all model updates
        for elem in data:
            file = io.BytesIO()
            file.write(elem)
            file.seek(0)
            content = np.load(file)

            model2, alpha2 = content['model'], content['alpha']

            total = self.alpha + alpha2
            model = (self.alpha * model + alpha2 * model2) / total
            self.alpha = total
        return model

    def _push_to_go(self, data):
        pass

    def _receive_from_go(self):
        return []
