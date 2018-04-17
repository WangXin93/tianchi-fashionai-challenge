from urllib.request import urlretrieve
import os
import tarfile
import sys

def main(root_dir):
    urls = [
        'https://en.myluo.cn/packages/fashionAI_attributes_test_a_20180222.tar',
        'https://en.myluo.cn/packages/fashionAI_attributes_train_20180222.tar',
        'https://en.myluo.cn/packages/warm_up_train_20180201.tar',
    ]

    for url in urls:
        filename = url.split('/')[-1]
        filename = os.path.join(root_dir, filename)
        urlretrieve(url, filename)

if __name__ == "__main__":
    root_dir = sys.argv[-1]
    main(root_dir)
