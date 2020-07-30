import pandas as pd
from pathlib import Path
from tqdm import tqdm

def csv_link_extrator(file_name):
    df = pd.read_csv(str(file_name))

    links = []
    subreddits = ['penis', 'cock', 'dicks',
                'averagepenis', 'MassiveCock', 'tinydick']
    urls = df[df['subreddit'].isin(subreddits)]['url'].tolist()
    urls = [u for u in urls if u.endswith('.jpg')]
    urls = list(set(urls))

    with open(file_name.stem+'.txt', 'w', encoding='utf-8') as f: 
        f.write('\n'.join(urls))

if __name__ == '__main__':
    csv_files = [f for f in Path('./').iterdir() if str(f).endswith('.csv')]
    for f in tqdm(csv_files):
        if not Path(f.stem + '.txt').exists():
            csv_link_extrator(f)
