import socket
import threading

# 32 vga
# 33 wb_con
# 34 sqrt
# 35 dynamic

class MessageForwardingServer:
    def __init__(self, host='127.0.0.1', port=65436):
        self.host = host
        self.port = port
        self.clients = []
        self.clients_lock = threading.Lock()  # 线程锁
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

    def handle_client(self, client_socket, client_address):
        with self.clients_lock:
            self.clients.append(client_socket)
        print(f"Connection from {client_address}")

        try:
            while True:
                try:
                    message = client_socket.recv(1024).decode('utf-8')
                    if not message:
                        break  # 客户端正常断开

                    # 转发消息给其他客户端
                    with self.clients_lock:
                        clients_copy = self.clients.copy()

                    for client in clients_copy:
                        if client != client_socket:
                            try:
                                client.sendall(message.encode('utf-8'))
                            except (ConnectionResetError, BrokenPipeError, OSError):
                                print(f"Client {client.getpeername()} 断开连接.")
                                with self.clients_lock:
                                    if client in self.clients:
                                        self.clients.remove(client)
                                client.close()
                except (ConnectionResetError, UnicodeDecodeError):
                    break  # 接收数据时发生异常
        finally:
            # 清理客户端
            with self.clients_lock:
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
            client_socket.close()
            print(f"{client_address} connection closed.")

    def start(self):
        while True:
            client_socket, client_address = self.server_socket.accept()
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_thread.start()

if __name__ == "__main__":
    server = MessageForwardingServer()
    server.start()