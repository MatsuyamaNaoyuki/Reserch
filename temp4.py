import ctypes

try:
    ctypes.WinDLL('nokovsdk.dll')
    print("nokovsdk.dll loaded successfully.")
except OSError as e:
    print(f"Failed to load nokovsdk.dll: {e}")