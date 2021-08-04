import pandas as pd

# get validation dataloader
val_dataset = pd.read_csv('data/validation.csv')
eval_results = pd.read_csv('results/multimodal_word_embed_random_init_val_results.csv')

# extract categories
val_category_dfs = val_dataset.groupby('target_category')

for category, category_df in val_category_dfs:
    html = ""

    # html start
    html += f"""
    <!doctype html>
    <html lang="en">
      <head>
        <!-- Required meta tags -->
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
     
        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
        <link rel="stylesheet" href="bootstrap-custom.css">
     
        <title>SAYCam Evaluation</title>
      </head>
      <body>
        <div class="container">
          <br />
          <h2>Category: {category.capitalize()}</h2>

          <br />
    """
    
    for index, row in category_df.iterrows():
        # extract row info
        target_category = row['target_category'].capitalize()
        foil_category_one = row['foil_category_one'].capitalize()
        foil_category_two = row['foil_category_two'].capitalize()
        foil_category_three = row['foil_category_three'].capitalize()
        target_img_filename = row['target_img_filename']
        foil_img_filename_one = row['foil_one_img_filename']
        foil_img_filename_two = row['foil_two_img_filename']
        foil_img_filename_three = row['foil_three_img_filename']
        target_img_sim = eval_results['target_sim'].iloc[index]
        foil_img_one_sim = eval_results['foil_one_sim'].iloc[index]
        foil_img_two_sim = eval_results['foil_two_sim'].iloc[index]
        foil_img_three_sim = eval_results['foil_three_sim'].iloc[index]

        # add target image
        html += f"""
        <div class="row">
            <div class="col-12">
                <h5>Validation Trial: {index}</h5>
            </div>
        </div>
        <div class="row">
            <div class="col-3">
              <figure class="figure">
                <img src="{'../'+target_img_filename}" class="figure-img img-fluid", style="width: 224px">
                <figcaption class="figure-caption text-center" style="width: 15rem"><strong>Target Image</strong> ({target_category})</figcaption>
              </figure>
            </div>
        """

        # add foil images
        html += f"""
            <div class="col-3">
              <figure class="figure">
                <img src="{'../'+foil_img_filename_one}" class="figure-img img-fluid", style="width: 224px">
                <figcaption class="figure-caption text-center" style="width: 15rem"><strong>Foil One</strong> ({foil_category_one})</figcaption>
              </figure>
            </div>
        """

        html += f"""
            <div class="col-3">
              <figure class="figure">
                <img src="{'../'+foil_img_filename_two}" class="figure-img img-fluid", style="width: 224px">
                <figcaption class="figure-caption text-center" style="width: 15rem"><strong>Foil Two</strong> ({foil_category_two})</figcaption>
              </figure>
            </div>
        """

        html += f"""
            <div class="col-3">
              <figure class="figure">
                <img src="{'../'+foil_img_filename_three}" class="figure-img img-fluid", style="width: 224px">
                <figcaption class="figure-caption text-center" style="width: 15rem"><strong>Foil Three</strong> ({foil_category_three})</figcaption>
              </figure>
            </div>
        </div>
        """

        # add heatmaps
        # add target image
        html += f"""
        <div class="row">
            <div class="col-3">
              <figure class="figure">
                <img src="{'../figures/multimodal_word_embed_random_init/trial_'+str(index)+'_target_heatmap.png'}" class="figure-img img-fluid", style="width: 224px">
                <figcaption class="figure-caption text-center" style="width: 15rem">Sim: {target_img_sim:02f}</figcaption>
              </figure>
            </div>
        """

        # add foil images
        html += f"""
            <div class="col-3">
              <figure class="figure">
                <img src="{'../figures/multimodal_word_embed_random_init/trial_' + str(index) + '_foil_one_heatmap.png'}" class="figure-img img-fluid", style="width: 224px">
                <figcaption class="figure-caption text-center" style="width: 15rem">Sim: {foil_img_one_sim:02f}</figcaption>
              </figure>
            </div>
        """

        html += f"""
            <div class="col-3">
              <figure class="figure">
                <img src="{'../figures/multimodal_word_embed_random_init/trial_'+str(index)+'_foil_two_heatmap.png'}" class="figure-img img-fluid", style="width: 224px">
                <figcaption class="figure-caption text-center" style="width: 15rem">Sim: {foil_img_two_sim:02f}</figcaption>
              </figure>
            </div>
        """

        html += f"""
            <div class="col-3">
              <figure class="figure">
                <img src="{'../figures/multimodal_word_embed_random_init/trial_'+str(index)+'_foil_three_heatmap.png'}" class="figure-img img-fluid", style="width: 224px">
                <figcaption class="figure-caption text-center" style="width: 15rem">Sim: {foil_img_three_sim:02f}</figcaption>
              </figure>
            </div>
        </div>
        """
        
        
    # html end
    html += f"""
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
    viz_filename = f'{category}.html'
    with open(f'eval_viz/{viz_filename}', 'w') as f:
        f.write(html)
