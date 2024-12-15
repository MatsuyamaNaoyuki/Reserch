import serial
import csv
import time
from datetime import datetime




arduino = serial.Serial('COM3', 115200, timeout=1)
arduino.reset_input_buffer()  # 入力バッファをクリア
arduino.reset_output_buffer()  # 出力バッファをクリア
# 現在の日時を取得してファイル名に埋め込む
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f'sensor_data_{current_time}.csv'            #fstringを用いて文字列に変数を埋め込む




if __name__ == '__main__':       #パッケージとして使ったときに使用しない．意図は不明
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time","air1", "air2", "air3", "air4"])
        starttime = time.perf_counter()
        try:
            while True:
                arduino.write(b'1')  # ここで値を送信

                if arduino.in_waiting > 0:#シリアルバッファに受信データがあるか
                    print("C")
                    #decodeはbyteをutf-8に変換する関数　rstrip()は末端の改行文字を削除
                    line = arduino.readline().decode('utf-8', errors='ignore').rstrip()
                    print(line)
                    data = line.split(',')
                    print(data)
                    if len(data) == 4:
                        try:
                            sd = [int(d) for d in data]
                            print(sd)
                            passtime = time.perf_counter() - starttime
                            writer.writerow([passtime] + sd)
                            print(f"Data written: {sd}")
                        except ValueError:
                            print("Received non-integer data.")

                time.sleep(0.05)  # 1秒間隔で読み取り要求
        except KeyboardInterrupt:           #Ctrl + Cを押すと止まるプログラム
            print("Program terminated")
        finally:
            arduino.close()
