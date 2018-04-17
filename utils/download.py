import os
import tarfile
import sys
import multiprocessing
import datatime


urls = [
    'https://en.myluo.cn/packages/fashionAI_attributes_test_a_20180222.tar',
    'https://en.myluo.cn/packages/fashionAI_attributes_train_20180222.tar',
    'https://en.myluo.cn/packages/warm_up_train_20180201.tar',
]


def download(url, root_dir='~/datasets'):
    """Download from url to root_dir
    

    Args:
        url: URL to be downloaded
        root_dir: Directory to store resource
    """
    print('Downloading {}...'.format(url))
    filename = url.split('/')[-1]
    filename = os.path.join(root_dir, filename)
    urlretrieve(url, filename)
    print('Completed {}...'.format(url))


if __name__ == "__main__":
    start = datetime.datetime.now()

    root_dir='~/datasets'
    processes = []
    for i in range(3):
        process = multiprocessing.Process(
            target=download, args=(urls[i], root_dir))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    end = datetime.datetime.now()
    print('The multiprocessed download loops took: %s' % (end-start))

