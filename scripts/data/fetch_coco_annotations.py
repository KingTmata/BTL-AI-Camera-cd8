"""Download official COCO2017 annotations only, never the image archives."""
import shutil
import urllib.request
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    target = root / "data/raw/coco500_v0.1/annotations"
    target.mkdir(parents=True, exist_ok=True)
    names = ["instances_train2017.json", "instances_val2017.json"]
    if all((target / name).is_file() for name in names):
        print("Annotation files already present", flush=True)
        return
    archive = target / "annotations_trainval2017.zip"
    if not archive.is_file():
        temporary = archive.with_suffix(".zip.part")
        url = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            total = 0
            last = 0
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                total += len(chunk)
                if total-last >= 20*1024*1024:
                    print(f"Annotations: {total//(1024*1024)} MiB downloaded", flush=True)
                    last = total
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        for name in names:
            if (target / name).is_file():
                continue
            with zipped.open("annotations/"+name) as source, (target / (name+".part")).open("wb") as stream:
                shutil.copyfileobj(source, stream)
            (target / (name+".part")).replace(target / name)
            print(f"Extracted {name}", flush=True)


if __name__ == "__main__":
    main()
