from pathlib import Path
import pandas as pd

from multimodal_saycam.data.base_data_module import BaseDataModule

PREPROCESSED_TRANSCRIPTS_DIRNAME = BaseDataModule.data_dirname() / "preprocessed_transcripts"
ANIMATED_FRAMES_DIRNAME = "img"

# get list of preprocessed transcripts
transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))[:5]

for idx, transcript in enumerate(transcripts):
    print(f'Creating visualization: {transcript} ({idx+1}/{len(transcripts)})')

    # read in preprocessed transcript
    transcript_df = pd.read_csv(transcript)
        
    # group by utterances
    utterance_groups = transcript_df.groupby('utterance_num')

    print(f'number of utterances in {transcript.stem}: {len(utterance_groups)}')

    html_string = ""

    # html start
    html_string += f"""
    <!doctype html>
    <html lang="en">
      <head>
        <!-- Required meta tags -->
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
     
        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
        <link rel="stylesheet" href="bootstrap-custom.css">
     
        <title>SAYCam</title>
      </head>
      <body>
        <div class="container">
          <h2>Video: {transcript.stem}</h2>

          <br />
          <div class="row">
    """

    # add html for each image/utterance pair
    for utterance_num, utterance_group in utterance_groups:
        utterance = pd.unique(utterance_group['utterance']).item()
        gif_filename = f"{pd.unique(utterance_group['transcript_filename']).item()[:-4]}_{utterance_num:03}.gif"
        gif_filepath = Path(ANIMATED_FRAMES_DIRNAME, gif_filename)
        
        html_string += f"""
            <div class="col-2">
              <figure class="figure">
                <img src="{gif_filepath}" class="figure-img img-fluid rounded">
                <figcaption class="figure-caption text-wrap" style="width: 15rem"><strong>utterance</strong>: {utterance}</figcaption>
              </figure>
            </div>
        """

    # html end
    html_string += f"""
          </div>

          <div class="d-flex justify-content-between">
              <div>Index: <a href="index.html">SAYCam Videos</a></div>
          </div>
        </div>
     
        <!-- Optional JavaScript -->
        <!-- jQuery first, then Popper.js, then Bootstrap JS -->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js" integrity="sha384-9/reFTGAW83EW2RDu2S0VKaIzap3H66lZH81PoYlFhbGU+6BZp6G7niu735Sk7lN" crossorigin="anonymous"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" integrity="sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV" crossorigin="anonymous"></script>
      </body>
    </html>
    """

    # save html to file
    viz_filename = transcript.stem + '.html'
    with open(f'viz/{viz_filename}', 'w') as f:
        f.write(html_string)
