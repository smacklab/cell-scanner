from collections import defaultdict
from ultralytics import YOLO
from PIL import Image
class WhiteBloodCellDetector:
    def __init__(self, detect_model_path, classify_model_path, device="cpu", DEBUG=False):
        self.CONFIDENCE_THRESHOLD = 0.25
        self.CLASSIFY_SIZE = 224
        self.DETECTION_SIZE = 512
        self.IMAGE_SIZE_RATIO_THRESHOLD = 0.7
        self.DEBUG = DEBUG
        self.device = device
        self.dmodel = YOLO(detect_model_path)
        self.cmodel = YOLO(classify_model_path)

    def is_gpu(self) -> bool:
        return self.device != "cpu"

    def detect(self, image: Image) -> defaultdict:
        wbcs = defaultdict(int)

        r = self.dmodel(image, device=self.device, verbose=False)[0] # results always a list of length 1

        if self.DEBUG:
            im_array = r.plot() # plot wbcs
            im = Image.fromarray(im_array[..., ::-1])
            im.show()  # show image

        if self.is_gpu():
            r.boxes = r.boxes.cpu()
        r.boxes = r.boxes.numpy()

        for conf, xywh in zip(r.boxes.conf, r.boxes.xywh):
            _, _, width, height = xywh
            if conf > self.CONFIDENCE_THRESHOLD and (self.IMAGE_SIZE_RATIO_THRESHOLD < width / height < 1 / self.IMAGE_SIZE_RATIO_THRESHOLD) :
                wbc_classname = self.classify(image, xywh)
                wbcs[wbc_classname] += 1
                
        return wbcs

    def classify(self, image: Image, xywh) -> str:
        center_x, center_y, _, _ = xywh
        left = max(center_x - self.CLASSIFY_SIZE // 2, 0)
        top = max(center_y - self.CLASSIFY_SIZE // 2, 0)
        right = min(center_x + self.CLASSIFY_SIZE // 2, self.DETECTION_SIZE)
        bottom = min(center_y + self.CLASSIFY_SIZE // 2, self.DETECTION_SIZE)
        cls_image = image.crop((left, top, right, bottom))

        r = self.cmodel(cls_image, device=self.device, verbose=False)[0]
        if self.is_gpu():
            r.boxes = r.boxes.cpu()
        r.boxes = r.boxes.numpy()

        cords = [(x,y,w,h,cls) for (x,y,w,h), cls in zip(r.boxes.xywh, r.boxes.cls)]
        closest = self.get_wbc_closest_to_center(cords)
        cls = closest[-1]
        return r.names[cls]
    
    def get_wbc_closest_to_center(self, cords: list) -> tuple:
        center = (self.CLASSIFY_SIZE // 2, self.CLASSIFY_SIZE // 2)
        distances = list(map(lambda cord: self.distance(cord, center), cords))
        closest_index = distances.index(min(distances))
        return cords[closest_index]
    
    def distance(self, p1: tuple, p2: tuple) -> float:
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    

class RedBloodCellDetector:
    def __init__(self, detect_model_path, device="cpu", DEBUG=False):
        self.model = YOLO(detect_model_path)
        self.CONFIDENCE_THRESHOLD = 0.25
        self.IMAGE_SIZE_RATIO_THRESHOLD = 0.7
        self.device = device
        self.DEBUG = DEBUG

    def is_gpu(self) -> bool:
        return self.device != "cpu"

    def detect(self, image: Image) -> int:
        rbc = 0

        r = self.model(image, device=self.device, verbose=False)[0] # results always a list of length 1

        if self.DEBUG:
            im_array = r.plot(labels=False) # plot rbcs
            im = Image.fromarray(im_array[..., ::-1])
            im.show()  # show image

        if self.is_gpu():
            r.boxes = r.boxes.cpu()
        r.boxes = r.boxes.numpy()

        for conf, xywh in zip(r.boxes.conf, r.boxes.xywh):
            _, _, width, height = xywh
            if conf > self.CONFIDENCE_THRESHOLD and (self.IMAGE_SIZE_RATIO_THRESHOLD < width / height < 1 / self.IMAGE_SIZE_RATIO_THRESHOLD) :
                rbc += 1
                
        return rbc
