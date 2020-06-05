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
    def __init__(self, update_queue):
        self.received_updates = update_queue

    def SendModel(self, request, context):
        """Call RPC Server's SendModel to send the model to RPC Server
        """
        logging.info("Received model")
        self.received_updates.put(request.data)
        return communication_pb2.Reply(result=True)


class ServerAPI:
    def __init__(self, peers, update_queue):
        self.peer_addrs = peers
        self.peers = [RpcClient(addr) for addr in self.peer_addrs]
        self.received_updates = update_queue

    def push_model(self, data):
        if len(self.peers) > 0:
            peer = random.choice(self.peers)
            logging.info("Sending to", peer.server_address)
            return peer.send_model(data)
        return False

    def get_updates(self):
        logging.info("Draining updates")
        for model in _drain(self.received_updates):
            yield model


def serve(port="50051", peers=None):
    if peers is None:
        peers = []
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    queue = Queue()
    server_api = ServerAPI(peers, queue)
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
    logging.basicConfig()
    me = get_ip() + ":50051"
    logging.info("Serving on", me)
    peers = read_peers(file_name, me)
    logging.info("Peers:", ", ".join(peers))
    return serve(port="50051", peers=peers)


def _drain(q):
    while True:
        try:
            yield q.get_nowait()
        except Empty:
            break
