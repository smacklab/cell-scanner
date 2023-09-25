
from typing import NamedTuple, Dict
from ultralytics import YOLO
from PIL import Image
import tifffile
import torch
from Detectors import WhiteBloodCellDetector, RedBloodCellDetector, BloodDensityDetector

# SETUP GPU
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)
###

ScanResult = NamedTuple("ScanResult", [("wbc", Dict), ("rbc", int)])

# def process__ndpi(ndpi):
#     wbc = {"N": 0, "L": 0, "M": 0, "E": 0, "B": 0}
#     rbc = 0
#     for h in ndpi_height:
#         for w in ndpi_width:
#             image = ndpi.crop(height, width, height + 512, width + 512)
#             process_image(image)


#     generate_summary(wbcs, rbc)

def process_image(image: Image) -> ScanResult:
    bloodDensityDetector = BloodDensityDetector("models/blood-smear-density-Apr10.pt", DEBUG=True)

    if not bloodDensityDetector.hasGoodDensity(image):
        # scan is not good, return empty result
        return ScanResult({}, 0)

    wbcDetector = WhiteBloodCellDetector("models/wbc-detection-Feb24.pt", "models/wbc-classification-Sep23.pt", DEBUG=True)
    rbcDetector = RedBloodCellDetector("models/rbc-detection-Sep12.pt", DEBUG=True)
    wbc = wbcDetector.detect(image)
    rbc = rbcDetector.detect(image)
    return ScanResult(wbc, rbc)


if __name__ == '__main__':

    cropped_image = Image.open("samples/sample1.jpg")
    print(process_image(cropped_image))
    # for ndpi in folder:
    #     process__ndpi(ndpi)
