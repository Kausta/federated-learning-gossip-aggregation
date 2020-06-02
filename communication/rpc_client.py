import grpc
from . import communication_pb2_grpc
from . import communication_pb2


class RpcClient:
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(self.server_address)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.channel.close()

    def send_model(self, data: bytes):
        stub = communication_pb2_grpc.CommunicatorStub(self.channel)
        response = stub.SendModel(communication_pb2.Model(data=data))

        return response

    def update_model(self, data: bytes):
        stub = communication_pb2_grpc.CommunicatorStub(self.channel)
        response = stub.SendModel(communication_pb2.Model(data=data))

        return response

    def receive_updates(self) -> [bytes]:
        stub = communication_pb2_grpc.CommunicatorStub(self.channel)
        response = stub.SendModel(communication_pb2.Model(data=data))

        return response
