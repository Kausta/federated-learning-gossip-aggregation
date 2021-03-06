from concurrent import futures
import logging
import random
import socket

import grpc

from .rpc_client import RpcClient
from . import communication_pb2
from . import communication_pb2_grpc
from queue import Queue, Empty


class Server(communication_pb2_grpc.CommunicatorServicer):
    """
    GRPC server.
    """
    def __init__(self, update_queue):
        self.received_updates = update_queue

    def PushModel(self, request, context):
        """
        Called by other peers to add model to received update queue. The received request is Model protobuf object
        """
        print("Received model")
        self.received_updates.put(request.data)
        return communication_pb2.Reply(result=True)


class ServerAPI:
    def __init__(self, peers, update_queue, server):
        self.peer_addrs = peers
        self.peers = [RpcClient(addr) for addr in self.peer_addrs]
        self.received_updates = update_queue
        self.server = server

    def push_model(self, data) -> bool:
        """
        If a peer exists, send to a peer, else return false
        :param data: Model update
        :return: Success bool
        """
        if len(self.peers) > 0:
            peer = random.choice(self.peers)
            print("Sending to", peer.server_address)
            return peer.send_model(data)
        return False

    def get_updates(self):
        """
        Returns received updates in the queue
        :return:
        """
        print("Draining updates")
        for model in _drain(self.received_updates):
            yield model


def serve(port="50051", peers=None):
    if peers is None:
        peers = []
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    queue = Queue()
    server_api = ServerAPI(peers, queue, server)
    communication_pb2_grpc.add_CommunicatorServicer_to_server(Server(queue), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()

    return server_api


def get_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]

    return ip


def read_peers(file_name, me):
    with open(file_name) as p:
        peers = p.readlines()
    peers = [peer.strip() for peer in peers]
    peers = [peer for peer in peers if peer != me]
    return peers


def server_from_peers_file(file_name):
    me = get_ip() + ":50051"
    print("Serving on", me)
    peers = read_peers(file_name, me)
    print("Peers:", ", ".join(peers))
    return serve(port="50051", peers=peers)


def _drain(q):
    while True:
        try:
            yield q.get_nowait()
        except Empty:
            break
