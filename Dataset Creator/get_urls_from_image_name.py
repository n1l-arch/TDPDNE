from pathlib import Path

if __name__ == '__main__':
    link_files = Path('dataset').iterdir()
    detector_images = Path('detector_images').iterdir()

    all_names = {}
    for l in link_files:
        with open(l) as f:
            links = f.read().split('\n')
            short_names = [u.split('/')[-1] for u in links]
            name_to_link = dict(zip(short_names, links))

        all_names.update(name_to_link)

    reddit_urls = []
    for d in detector_images:
        reddit_urls.append(all_names[d.name])

    with open('detector_urls.txt', 'w') as f:
        f.write('\n'.join(reddit_urls))
