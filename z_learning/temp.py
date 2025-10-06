
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# デバッグ
print("sys.path head:", sys.path[:3])
from myclass import MyModel
print("LOADED:", MyModel.__file__)

