motorlen = 3
maglen = 9
motionlen = 5

lenname = ["time"] +\
          [f"rotate{i}" for i in range(1, motorlen+1)] + \
          [f"force{i}"  for i in range(1, motorlen+1)] + \
          [f"sensor{i}"  for i in range(1, maglen+1)] + \
          [f"Mc{i}{p}" for i in range(1, motionlen+1) for p in ("x", "y", "z")]

print(lenname)
