import sys
from PIL import Image
import tifffile
from Detectors import WhiteBloodCellDetector, RedBloodCellDetector, BloodDensityDetector, ScanResult
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def process_chunk(ndpi, height, width):
    croppedImage = ndpi.crop((height, width, height + 512, width + 512))
    result = process_image(croppedImage)
    return result


def parallel_process(ndpi, ndpiHeight, ndpiWidth):
    summary = ScanResult()
    summaryLog = tqdm(total=0, position=2, bar_format='{desc}')

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for height in range(0, ndpiHeight, 512):
            for width in range(0, ndpiWidth, 512):
                future = executor.submit(process_chunk, ndpi, height, width)
                futures.append(future)

        for future in tqdm(futures, position=0, leave=True):
            result = future.result()
            summary.wbc += result.wbc
            summary.rbc += result.rbc
            summaryLog.set_description_str(f'{summary}')

    return summary


def process_ndpi(ndpiFile: str, save: bool) -> ScanResult:
    if not os.path.isfile(ndpiFile) or not ndpiFile.endswith(".ndpi"):
        print("Invalid NDPI file")
        return ScanResult()  # empty result

    summary = ScanResult()
    print("Reading NDPI Scan (approx. 15 seconds)")

    with tifffile.TiffFile(ndpiFile) as tif:
        ndpiRaw = tif.asarray()
        print("Done reading, start processing NDPI Scan")
        ndpi = Image.fromarray(ndpiRaw)
        ndpiWidth, ndpiHeight = ndpi.size

        print("Processing row by row, column by column (approx. 5 minutes on GPU)")
        summary = parallel_process(ndpi, ndpiHeight, ndpiWidth)

    if save:
        # print summary to ~/results/filename.txt
        with open(os.path.expanduser("~/results/" + os.path.basename(ndpiFile) + ".txt"), "w") as f:
            f.write(str(summary))

    return summary


def process_image(image: Image) -> ScanResult:
    bloodDensityDetector = BloodDensityDetector("models/blood-smear-density-Apr10.pt")

    if not bloodDensityDetector.hasGoodDensity(image):
        # scan is not good, return empty result
        return ScanResult({}, 0)

    wbcDetector = WhiteBloodCellDetector("models/wbc-classification-Sep23.pt", DEBUG=False)
    rbcDetector = RedBloodCellDetector("models/rbc-detection-Sep12.pt", DEBUG=False)
    wbc = wbcDetector.detect(image)
    rbc = rbcDetector.detect(image)

    return ScanResult(wbc, rbc)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python main.py ndpi <pathToNdpiFile>")
        sys.exit(1)

    if sys.argv[1] != "ndpi":
        print("Usage: python main.py ndpi <pathToNdpiFile>")
        sys.exit(1)

    ndpiFile = sys.argv[2]
    summary = process_ndpi(ndpiFile, save=True)

    print(summary)
