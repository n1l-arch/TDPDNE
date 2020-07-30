import lzma
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd


def xz_to_json(file_name):
    rows = []

    try:
        with lzma.open(str(file_name), mode='rt') as f:
            for line in tqdm(f, total=12000000):
                data = json.loads(line)
                relevant_data = {
                    'subreddit': data['subreddit'], 'title': data['title'], 'url': data['url']}
                rows.append(relevant_data)

    except Exception as e:
        print(e)
        pass
    finally:
        pd.DataFrame(rows).to_csv(f'{file_name.stem}.csv', index=False)


if __name__ == '__main__':
    compressed_files = [f for f in Path('./').iterdir() if str(f).endswith('.xz')]
    for f in tqdm(compressed_files):
        if not Path(f.stem + '.csv').exists():
            xz_to_json(f)
