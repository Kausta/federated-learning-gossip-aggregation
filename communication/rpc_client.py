import grpc
import communication_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = communication_pb2_grpc.CommunicatorStub(channel)