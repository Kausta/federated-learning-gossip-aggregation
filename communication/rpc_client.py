import grpc
import communication_pb2_grpc
import communication_pb2

SERVER_ADDRESS = 'localhost:50051'

def send_model(data: bytes):
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        stub = communication_pb2_grpc.CommunicatorStub(channel)
        response = stub.SendModel(communication_pb2.Model(data=data))

    return response

def get_updates():
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        stub = communication_pb2_grpc.CommunicatorStub(channel)
        response = stub.ReceiveUpdates(communication_pb2.Reply())

    print(response)
    return response


if __name__ == '__main__':
    send_model(bytes(3))
    send_model(bytes(1))
    send_model(bytes(7))
    send_model(bytes(9))
    get_updates()
