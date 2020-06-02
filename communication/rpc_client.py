import grpc
from . import communication_pb2_grpc
from . import communication_pb2


class RpcClient:
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address

    def send_model(self, data: bytes):
        with grpc.insecure_channel(self.server_address) as channel:
            stub = communication_pb2_grpc.CommunicatorStub(channel)
            response = stub.SendModel(communication_pb2.Model(data=data))

        return response

    def get_updates(self) -> [bytes]:
        with grpc.insecure_channel(self.server_address) as channel:
            stub = communication_pb2_grpc.CommunicatorStub(channel)
            response = stub.ReceiveUpdates(communication_pb2.Reply())

        return response.data
