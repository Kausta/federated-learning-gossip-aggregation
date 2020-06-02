from concurrent import futures
import logging

import grpc

import communication_pb2
import communication_pb2_grpc
from queue import Queue, Empty


class Server(communication_pb2_grpc.CommunicatorServicer):
    def __init__(self, peers):
        self.peers = peers

    received_updates = Queue()
    sending_updates = Queue()

    def SendModel(self, request, context):
        """Call RPC Server's SendModel to send the model to RPC Server
        """
        self.received_updates.put(request.data)
        return communication_pb2.Reply(result="Success")

    def UpdateModel(self, request, context):
        """Gets model from python and routes to a peer. Should queue if no peer exists
        """
        if len(self.peers) > 0:
            print("HEY")
        self.sending_updates.put(request.data)
        return communication_pb2.Reply(result="Success")

    def ReceiveUpdates(self, request, context):
        """Send update. TODO: change to streaming to process efficiently in the future
        """
        result = communication_pb2.Models()
        for model in _drain(self.received_updates):
            result.data.append(model)
            print(model)
        return result


def serve(port="50051", peers=None, sync=False):
    if peers is None:
        peers = []
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    communication_pb2_grpc.add_CommunicatorServicer_to_server(Server(peers), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()

    if sync:
        server.wait_for_termination()   # Comment this out for non-waiting start


def _drain(q):
    while True:
        try:
            yield q.get_nowait()
        except Empty:
            break


if __name__ == '__main__':
    logging.basicConfig()
    serve(sync=True)
