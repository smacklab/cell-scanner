
from PIL import Image
import tifffile
import torch
from Detectors import WhiteBloodCellDetector, RedBloodCellDetector, BloodDensityDetector, ScanResult
import os
from tqdm import tqdm


# SETUP GPU
device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)
###


def process_ndpi(ndpiFile: str) -> ScanResult:
    if not os.path.isfile(ndpiFile) or not ndpiFile.endswith(".ndpi"):
        print("Invalid NDPI file")
        return ScanResult()

    summary = ScanResult()
    print("Reading NDPI Scan (approx. 15 seconds)")

    with tifffile.TiffFile(ndpiFile) as tif:
        ndpiRaw = tif.asarray()
        print("Done reading, start processing NDPI Scan")
        ndpi = Image.fromarray(ndpiRaw)
        ndpiWidth, ndpiHeight = ndpi.size
        summaryLog = tqdm(total=0, position=2, bar_format='{desc}')

        for height in tqdm(range(0, ndpiHeight, 512), position=0, leave=False, desc="Rows"):
            for width in tqdm(range(0, ndpiWidth, 512), position=1, leave=False, desc="Columns"):
                croppedImage = ndpi.crop((height, width, height + 512, width + 512))
                result = process_image(croppedImage)
                # combine results
                summary.wbc += result.wbc
                summary.rbc += result.rbc
                summaryLog.set_description_str(f'{summary}')

    return summary


def process_image(image: Image) -> ScanResult:
    bloodDensityDetector = BloodDensityDetector("models/blood-smear-density-Apr10.pt")

    if not bloodDensityDetector.hasGoodDensity(image) and False:
        # scan is not good, return empty result
        return ScanResult({}, 0)

    wbcDetector = WhiteBloodCellDetector("models/wbc-detection-Feb24.pt", "models/wbc-classification-Sep23.pt", device=device)
    rbcDetector = RedBloodCellDetector("models/rbc-detection-Sep12.pt", device=device)
    wbc = wbcDetector.detect(image)
    rbc = rbcDetector.detect(image)

    return ScanResult(wbc, rbc)


if __name__ == '__main__':

    ndpi_file = "samples/[F012]2019-107_MID.ndpi"
    summary = process_ndpi(ndpi_file)

    print(summary)
