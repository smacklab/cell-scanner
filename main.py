
from PIL import Image
import tifffile
from Detectors import WhiteBloodCellDetector, RedBloodCellDetector, BloodDensityDetector, ScanResult
import os
from tqdm import tqdm


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
        summaryLog = tqdm(total=0, position=2, bar_format='{desc}')
        for height in tqdm(range(0, ndpiHeight, 512), position=0):
            for width in tqdm(range(0, ndpiWidth, 512), leave=False, position=1):
                croppedImage = ndpi.crop((height, width, height + 512, width + 512))
                result = process_image(croppedImage)  # Extract WBC and RBC data from cropped image

                # combine results
                summary.wbc += result.wbc
                summary.rbc += result.rbc
                summaryLog.set_description_str(f'{summary}')

    if save:
        # print summary to file
        with open(os.path.splitext(ndpiFile)[0] + ".txt", "w") as f:
            f.write(str(summary))

    return summary


def process_image(image: Image) -> ScanResult:
    bloodDensityDetector = BloodDensityDetector("models/blood-smear-density-Apr10.pt")

    # if not bloodDensityDetector.hasGoodDensity(image):
    # scan is not good, return empty result
    # return ScanResult({}, 0)

    wbcDetector = WhiteBloodCellDetector("models/wbc-classification-Sep23.pt", DEBUG=False)
    rbcDetector = RedBloodCellDetector("models/rbc-detection-Sep12.pt", DEBUG=False)
    wbc = wbcDetector.detect(image)
    # rbc = rbcDetector.detect(image)

    return ScanResult(wbc, 0)


if __name__ == '__main__':
    summary = ScanResult()

    # open file samples/Set5/active.txt
    # the file contains image names of the active samples
    # each line is a image name
    # for each filename, run the process_image function
    summaryLog = tqdm(total=0, position=0, bar_format='{desc}')
    for filename in tqdm(open("samples/set5active.txt").readlines()):
        filename = filename.strip()
        image = Image.open("samples/Set5/" + filename)
        result = process_image(image)
        summary.wbc += result.wbc
        summary.rbc += result.rbc
        summaryLog.set_description_str(f'{summary}')

    print(summary)
