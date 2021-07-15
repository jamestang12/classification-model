import zipfile
import os


def test(data_dir):
    for item in os.listdir(data_dir):
            if item.endswith(".zip"):
                path = data_dir + "/" + item
                print(item)
                print(path)
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    newDir = data_dir + "/zip"
                    os.makedirs(newDir)
                    zip_ref.extractall(newDir)
                    print(os.listdir(newDir + "/images"))

print(os.getcwd())
test(os.getcwd())