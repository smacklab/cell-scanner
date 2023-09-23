from typing import Dict
from ultralytics import YOLO
from PIL import Image
import tifffile
import torch
torch.cuda.set_device(0)


def process__ndpi(ndpi):
    wbc = {"N": 0, "L": 0, "M": 0, "E": 0, "B": 0}
    rbc = 0
    for h in ndpi_height:
        for w in ndpi_width:
            image = ndpi.crop(height, width, height + 512, width + 512)
            wbc = wbc | count_wbcs(image)  # merge 2 dictionaries
            rbc += count_rbc(image)

    generate_summary(wbcs, rbc)


if __name__ == '__main__':
    for ndpi in folder:
        process__ndpi(ndpi)


class WhiteBloodCellDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def is_gpu(self) -> bool:
        return self.model.device.type == "cuda"

    def count(self, image: Image) -> Dict:
        results = self.model(image)
        for r in results:
            # if gpu then convert to cpu
            if self.is_gpu():
                r.boxes = r.boxes.cpu()
