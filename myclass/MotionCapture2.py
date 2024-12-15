import socket
import struct
import sys,os, time, datetime, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class MotionCapture():
    def __init__(self):
        # マルチキャストグループのIPアドレスとポート番号を設定
        # うまくいかないときはMotionCaptureの設定を見直すこと
        self.MCAST_GRP = '239.239.239.52'
        self.MCAST_PORT = 5231
        # ソケットを作成
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # 受信バッファのサイズを設定
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # ソケットをバインド
        self.sock.bind(('', self.MCAST_PORT))

        # マルチキャストグループに参加
        self.mreq = struct.pack("4sl", socket.inet_aton(self.MCAST_GRP), socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, self.mreq)
        self.datas = []




    def get_data(self, timereturn = True):
        # ノンブロッキングモードに設定
        self.sock.setblocking(False)
        # 受信バッファサイズを拡大
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        clear_udp_buffer(self.sock)
        self.sock.setblocking(True)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        now_time  = datetime.datetime.now()
        print("X")
        data, addr = self.sock.recvfrom(1024)
        formatted_now = now_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # print(data)
        if timereturn == True:
            return data, formatted_now
        else:
            return data
    
    def store_data(self, data, now_time):
        byte_len = len(data)
        floatdatas = []
        for i in range(28,byte_len - 40, 4):
            byte_chunk = data[i:i+4]
            if len(byte_chunk) < 4:
                continue
            float_value = struct.unpack('f', byte_chunk)[0]
            floatdatas.append(float_value)
        floatdatas.insert(0, now_time)
        print(floatdatas)
        self.datas.append(floatdatas)
    
    def change_data(self, data):
        byte_len = len(data)
        floatdatas = []
        for i in range(28,byte_len - 40, 4):
            byte_chunk = data[i:i+4]
            if len(byte_chunk) < 4:
                continue
            float_value = struct.unpack('f', byte_chunk)[0]
            floatdatas.append(float_value)
        return floatdatas


        
# バッファをクリアする関数
def clear_udp_buffer(sock, timeout=1):
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            data, addr = sock.recvfrom(65535)
            # print("delete__")  # デバッグのために表示（実際には不要）
        except BlockingIOError:
            # バッファが空になったら抜ける
            break
        
        
# mc = MotionCapture()

# while(True):
#     data,nowtime = mc.get_data()
#     mc.store_data(data, nowtime)
# data,nowtime = mc.get_data()
# mc.store_data(data, nowtime)
# print(mc.datas)
# filename = 'test'
# now = datetime.datetime.now()
# filename = os.path.dirname(__file__) +"\\" + filename + now.strftime('%Y%m%d_%H%M%S') + '.csv'

# with open(filename, 'w',newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(mc.datas)

