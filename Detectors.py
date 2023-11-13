from IPython.display import display
from collections import Counter
from ultralytics import YOLO
from PIL import Image
import sys
from dataclasses import dataclass, field
import torch


@dataclass
class ScanResult:
    wbc: Counter = field(default_factory=Counter)
    rbc: int = 0


class Singleton(type):
    # https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class WhiteBloodCellDetector(metaclass=Singleton):
    def __init__(self, classify_model_path, DEBUG=False):
        self.CONFIDENCE_THRESHOLD = {"L": 0.5, "M": 0.25, "N": 0.1, "E": 0.8, "B": 0.1}
        self.DETECTION_SIZE = 1024
        self.IMAGE_SIZE_RATIO_THRESHOLD = 0.5
        self.DEBUG = DEBUG
        self.DEVICE = "0" if torch.cuda.is_available() else "cpu"
        self.cmodel = YOLO(classify_model_path)

    def is_gpu(self) -> bool:
        return self.DEVICE != "cpu"

    def detect(self, image: Image) -> Counter:
        wbcs = Counter()

        r = self.cmodel(image, device=self.DEVICE, imgsz=self.DETECTION_SIZE, verbose=False)[0]  # results always a list of length 1

        if self.DEBUG:
            im_array = r.plot()  # plot wbcs
            im = Image.fromarray(im_array[..., ::-1])
            if 'ipykernel' in sys.modules:
                display(im)  # show image
            else:
                im.show()

        if self.is_gpu():
            r.boxes = r.boxes.cpu()
        r.boxes = r.boxes.numpy()
        for conf, xywh, cls in zip(r.boxes.conf, r.boxes.xywh, r.boxes.cls):
            _, _, width, height = xywh
            wbc_classname = r.names[cls]
            if conf > self.CONFIDENCE_THRESHOLD[wbc_classname] and (self.IMAGE_SIZE_RATIO_THRESHOLD < width / height < 1 / self.IMAGE_SIZE_RATIO_THRESHOLD):
                wbcs[wbc_classname] += 1

        return wbcs


class RedBloodCellDetector(metaclass=Singleton):
    def __init__(self, detect_model_path, DEBUG=False):
        self.CONFIDENCE_THRESHOLD = 0.4
        self.IMAGE_SIZE_RATIO_THRESHOLD = 0.7
        self.DEVICE = "0" if torch.cuda.is_available() else "cpu"
        self.DEBUG = DEBUG
        self.model = YOLO(detect_model_path)

    def is_gpu(self) -> bool:
        return self.DEVICE != "cpu"

    def detect(self, image: Image) -> int:
        rbc = 0

        r = self.model(image, device=self.DEVICE, verbose=False)[0]  # results always a list of length 1

        if self.DEBUG:
            im_array = r.plot(font_size=0.01, line_width=1)  # plot rbcs
            im = Image.fromarray(im_array[..., ::-1])
            if 'ipykernel' in sys.modules:
                display(im)  # show image in Jupyter Notebook
            else:
                im.show()  # show image

        if self.is_gpu():
            r.boxes = r.boxes.cpu()
        r.boxes = r.boxes.numpy()

        for conf, xywh in zip(r.boxes.conf, r.boxes.xywh):
            _, _, width, height = xywh
            if conf > self.CONFIDENCE_THRESHOLD and height > 0 and (self.IMAGE_SIZE_RATIO_THRESHOLD < width / height < 1 / self.IMAGE_SIZE_RATIO_THRESHOLD):
                rbc += 1

        return rbc


class BloodDensityDetector(metaclass=Singleton):
    def __init__(self, density_model_path, DEBUG=False):
        self.model = YOLO(density_model_path)
        self.DEBUG = DEBUG

    def hasGoodDensity(self, image: Image) -> int:
        r = self.model(image, verbose=False)[0]  # results always a list of length 1

        if self.DEBUG:
            im_array = r.plot(labels=False)  # plot density
            im = Image.fromarray(im_array[..., ::-1])
            if 'ipykernel' in sys.modules:
                display(im)  # show image in Jupyter Notebook
            else:
                im.show()

        cls_idx = r.probs.top1
        cls_name = r.names[cls_idx]

        return cls_name == "Good"
