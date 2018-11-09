import cntk as C
from pprint import pprint
import numpy as np
from imageio import imread
from skimage.transform import resize
from skimage.color import rgb2grey


fp = "C:/Users/delzac/OneDrive/Work/SAFORO Inter" \
     "nal Training/Deep learning with CNTK/Pre-train" \
     "ed models/EmotionFER/model.onnx"

model = C.load_model(fp, format=C.ModelFormat.ONNX)
C.logging.plot(model, 'emotionfer.png')
print(model)

# image_fp = "C:/Users/delzac/OneDrive/Work/Analytics Presentation/Demo/imageclassification/datasets/images/military/RSAF50_F15SG.jpg"
# image = imread(image_fp)
# image = resize(image, (224, 224))
# image = np.moveaxis(image, -1, 0)[None, ...]
# result = model.eval({model.arguments[0]: image})
# print(result[0].argmax())
