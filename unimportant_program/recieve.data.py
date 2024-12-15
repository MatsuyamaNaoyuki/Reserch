import socket

# 受信するポート番号
PORT = 5231
# ブロードキャストアドレス
BROADCAST_ADDR = "239.239.239.52"

# ソケットを作成
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ソケットオプションの設定
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# ポートにバインド
sock.bind(('', PORT))

print(f"Listening for broadcast messages on port {PORT}")

while True:
    print("data")
    data, addr = sock.recvfrom(1024)
    print("data")
    print(addr)
    print(f"Received message from {addr}: {data.decode()}")