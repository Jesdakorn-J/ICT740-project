from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_edgetpu_model.py model_edgetpu.tflite")
        sys.exit(1)

    model_path = sys.argv[1]

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    w, h = input_size(interpreter)
    print("Model loaded successfully.")
    print(f"Input size: {w}x{h}")

if __name__ == "__main__":
    main()
