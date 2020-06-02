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
        response = self.stub.SendModel(communication_pb2.Model(data=data))

        return response

    def update_model(self, data: bytes):
        response = self.stub.UpdateModel(communication_pb2.Model(data=data))

        return response

    def receive_updates(self) -> [bytes]:
        response = self.stub.ReceiveUpdates(communication_pb2.Reply(result="Success"))

        return response
