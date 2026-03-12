import socket
import time
import select
import random

def manage_socket(ip, port):
    try:
        # 创建 socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, port))
        client_socket.setblocking(False)  # 设置为非阻塞模式
        print(f"Connected to server {ip}:{port}")
        return client_socket
    except Exception as e:
        print(f"Error managing socket: {e}")
        return None

def send_data(sock, data):
    try:
        sock.sendall(data.encode())
        # print(f"client Sent: {data}")
    except Exception as e:
        print(f"Failed to send data: {e}")
def receive_data(sock, timeout=2000):
    try:
        ready_to_read, _, _ = select.select([sock], [], [], timeout)
        if ready_to_read:
            data = sock.recv(1024).decode()
            if data:
                # print(f"client Received: {data}")
                return data
            else:
                print("Connection closed by the server")
                return None
        else:
            print("No data available (timeout)")
            return None
    except Exception as e:
        print(f"Failed to receive data: {e}")
        return None