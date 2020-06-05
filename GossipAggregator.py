import abc
import io
import numpy as np
from communication.rpc_server import ServerAPI


# Decay Rate = 1 => equivalent to original averaging algorithm, no alpha update
# Decay Rate = 0 => equivalent to always adding same alpha update
# In between => added alpha update gets smaller by time


class GossipAggregator:
    def __init__(self, data_points, decay_rate, server_api: ServerAPI):
        self.alpha = 0
        self.alpha_update = float(data_points) / 10000
        self.decay_rate = decay_rate

        self.client = server_api

    def reset_update_rate(self, new_update_rate):
        self.alpha_update = new_update_rate

    def push_model(self, model):
        # Update alpha
        prev_alpha = self.alpha

        self.alpha += self.alpha_update
        self.alpha_update *= (1 - self.decay_rate)

        print("Alpha:", prev_alpha, "->", self.alpha, "->", self.alpha / 2)

        self.alpha /= 2
        # Compress to byte array
        file = io.BytesIO()
        np.savez_compressed(file, model=model, alpha=self.alpha)
        data = file.getbuffer()
        # Send
        res = self.client.push_model(data.tobytes())
        if not res:
            self.alpha *= 2
            print("Failed transmission, restoring alpha to", self.alpha)

    def receive_updates(self, model):
        # Process all model updates
        for elem in self.client.get_updates():
            file = io.BytesIO()
            file.write(elem)
            file.seek(0)
            content = np.load(file)

            model2, alpha2 = content['model'], content['alpha']

            total = self.alpha + alpha2
            print("Alpha:", self.alpha, "->", total)

            model = (self.alpha * model + alpha2 * model2) / total
            self.alpha = total
        return model
