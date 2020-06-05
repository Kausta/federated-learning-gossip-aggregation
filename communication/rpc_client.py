import logging

import grpc
from . import communication_pb2_grpc
from . import communication_pb2


class RpcClient:
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = communication_pb2_grpc.CommunicatorStub(self.channel)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.channel.close()

    def send_model(self, data: bytes):
        try:
            response = self.stub.PushModel(communication_pb2.Model(data=data))
            return response.result
        except grpc.RpcError as e:
            logging.error(e)
            return False
