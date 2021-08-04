# script to download SAYcam transcripts

import time
import pandas as pd
from gsheets import Sheets

# set up ghseets object
sheets = Sheets.from_files('credentials.json')

# get urls of saycam files to download
df = pd.read_csv('data/SAYcam_transcript_links.csv')
urls = df['GoogleSheets Link'].unique()[64:65]

for i, url in enumerate(urls):
    print(f'saving SAYcam sheet {i+1}/{len(urls)}: {url}')
    s = sheets.get(url)
    title = s.title.split('_')
    title = '_'.join(title[:3])

    for j in range(1, len(s.sheets)):
        df = s.sheets[j].to_frame()  # convert worksheet to data frame
        filename = f'data/transcripts/{title}_{s.sheets[j].title}.csv'  # get filename of dataframe
        df.to_csv(filename, index=False)  # save as dataframe

    # sleep for 5 seconds to prevent rate limiting
    time.sleep(30)
