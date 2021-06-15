import pandas as pd
from dataset import SAYCamTrainDataset

dataset = train_dataset = SAYCamTrainDataset()
transcripts = dataset.transcripts
video_filenames = sorted(transcripts['video_filename'].unique())

for idx, video_filename in enumerate(video_filenames):
    # extract rows for single video
    single_transcript = transcripts[transcripts['video_filename'] == video_filename]
     
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
          <h2>Video: {video_filename}</h2>
          <div class="d-flex justify-content-between">
              <div>Previous Video: <a href="{video_filenames[idx-1][:-4]}.html">{video_filenames[idx-1]}</a></div>
              <div>Index: <a href="index.html">SAYCam Videos</a></div>
              <div>Next Video: <a href="{video_filenames[(idx+1) % (len(video_filenames))][:-4]}.html">{video_filenames[(idx+1) % (len(video_filenames))]}</a></div>
          </div>

          <br />
          <div class="row">
    """

    # add html for each image/utterance pair
    for i in range(len(single_transcript)):
        html_string += f"""
            <div class="col-2">
              <figure class="figure">
                <img src="img/{single_transcript['frame_filename'].iloc[i]}" class="figure-img img-fluid rounded">
                <figcaption class="figure-caption text-wrap" style="width: 15rem"><strong>time</strong>: {single_transcript['original_time'].iloc[i]}, <strong>utterance</strong>: {single_transcript['utterance'].iloc[i]}</figcaption>
              </figure>
            </div>
        """

    # html end
    html_string += f"""
          </div>

          <div class="d-flex justify-content-between">
              <div>Previous Video: <a href="{video_filenames[idx-1][:-4]}.html">{video_filenames[idx-1]}</a></div>
              <div>Index: <a href="index.html">SAYCam Videos</a></div>
              <div>Next Video: <a href="{video_filenames[(idx+1) % (len(video_filenames))][:-4]}.html">{video_filenames[(idx+1) % (len(video_filenames))]}</a></div>
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
    viz_filename = f'{video_filename[:-4] + ".html"}'
    with open(f'viz/{viz_filename}', 'w') as f:
        f.write(html_string)
