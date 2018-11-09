import cntk as C
from cntk.layers import Convolution2D, MaxPooling, Dense, BatchNormalization, Dropout
from cntk.layers import ResNetBlock

fp = "C:/Users/delzac/OneDrive/Work/SAFORO Internal Training/" \
     "Deep learning with CNTK/Pre-trained models/SqueezeNet/squeezenet1.1.onnx"
model = C.load_model(fp, format=C.ModelFormat.ONNX)

print(model)
C.logging.plot(model, "model.png")
