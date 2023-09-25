
from PIL import Image
import tifffile
import torch
from Detectors import WhiteBloodCellDetector, RedBloodCellDetector, BloodDensityDetector, ScanResult
import os
from tqdm import tqdm_gui as tqdm


# SETUP GPU
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)
###


def process_ndpi(ndpiFile: str) -> ScanResult:
    if not os.path.isfile(ndpiFile) or not ndpiFile.endswith(".ndpi"):
        print("Invalid NDPI file")
        return ScanResult({}, 0)

    summary = ScanResult({}, 0)

    with tifffile.TiffFile(ndpiFile) as tif:
        ndpiRaw = tif.asarray()
        ndpi = Image.fromarray(ndpiRaw)
        ndpiWidth, ndpiHeight = ndpi.size
        for height in tqdm(range(0, ndpiHeight, 512)):
            for width in tqdm(range(0, ndpiWidth, 512), leave=False):
                croppedImage = ndpi.crop((height, width, height + 512, width + 512))
                result = process_image(croppedImage)
                # combine results
                summary.wbc.update(result.wbc)
                summary.rbc += result.rbc

    return summary


def process_image(image: Image) -> ScanResult:
    bloodDensityDetector = BloodDensityDetector("models/blood-smear-density-Apr10.pt")

    if not bloodDensityDetector.hasGoodDensity(image) and False:
        # scan is not good, return empty result
        return ScanResult({}, 0)

    wbcDetector = WhiteBloodCellDetector("models/wbc-detection-Feb24.pt", "models/wbc-classification-Sep23.pt")
    rbcDetector = RedBloodCellDetector("models/rbc-detection-Sep12.pt")
    wbc = wbcDetector.detect(image)
    rbc = rbcDetector.detect(image)

    return ScanResult(wbc, rbc)


if __name__ == '__main__':

    ndpi_file = "samples/[F001]2017-15_MID.ndpi"
    print("Reading NDPI Scan")
    summary = process_ndpi(ndpi_file)

    print(summary)
    # for ndpi in folder:
    #     process__ndpi(ndpi)
