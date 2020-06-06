import abc
import io
import numpy as np
from communication.rpc_server import ServerAPI


class GossipAggregator:
    def __init__(self, data_points, server_api: ServerAPI):
        self.alpha = float(data_points) / 10000

        self.client = server_api

    def push_model(self, model):
        """
        Updates the alpha and sends the model to a peer. Will restore the original alpha
        if the transmission of update fails for some reason.
        :param model:
        :return:
        """
        # Update alpha
        print("Alpha:", self.alpha, "->", self.alpha / 2)

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
        """
        Processes all received updates
        :param model:
        :return:
        """
        for elem in self.client.get_updates():
            # Write the bytes into memory for numpy to load from.
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
