import logging

from communication.rpc_server import serve

import socket


def get_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]

    return ip


if __name__ == '__main__':
    logging.basicConfig()
    me = get_ip() + ":50051"
    print("Serving on", me)
    with open('peers.txt') as p:
        peers = p.readlines()
    peers = [peer.strip() for peer in peers]
    peers = [peer for peer in peers if peer != me]
    print("Peers:", ", ".join(peers))
    serve(port="50051", peers=peers, sync=True)
