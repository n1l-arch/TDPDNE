import praw
import requests

reddit = praw.Reddit(client_id='SpqwRv8bs2Hf5g',
                     client_secret='m9JVG1I1tqlW1ZZakQNKmICbanc',
                     username='cyonb',
                     password='suman634',
                     user_agent='SportsStreams')

subreddit = reddit.subreddit('dicks')

for submission in subreddit.top('all'):
    with open('submissions.txt', 'a') as f:
        f.write(submission.url + '\n')
