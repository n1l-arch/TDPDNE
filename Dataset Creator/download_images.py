import urllib.request as req
from urllib.error import HTTPError, URLError
from tqdm import tqdm
from multiprocessing import Pool
from http.client import InvalidURL


def get_picture(url_and_file_num):
    url, file_num = url_and_file_num[0], url_and_file_num[1]
    file_name = url.strip('\n').split('/')[-1]

    try:
        req.urlretrieve(url, f'images/{file_name}')
    except (HTTPError, URLError, InvalidURL):
        pass

    return


if __name__ == '__main__':
    with open('RS_2018-05.txt', encoding='utf-8') as f:
        links = f.readlines()

    links = list(zip(links, range(len(links))))

    # for l in tqdm(links):
    #     get_picture(l)

    with Pool(10) as p:
        for _ in tqdm(p.imap(get_picture, links), total=len(links)):
            pass
