# ECS 171 Group 7 Project

Welcome to Group 7’s repository for the ECS 171 group project. Currently, Spotify has a limited selection of kids' tracks because they seem to be hand selected by Spotify employees. Our project aims to create a classifier that takes in the song lyrics and predicts whether the song is appropriate for kids or inappropriate for kids. We looked at various NLP and machine learning techniques to best figure out how the data should be processed and what classifier we should use. What you will find in this repository is our experimentation and the final model we decided to deploy to our frontend. 

## Data Creation

In this sub repository you will find all the scripts that were used in the creation of the data set. These scripts do not need to be run in order for the models to be built. The assembled data set is in the data sub repository. Although there are various `.csv` files, the final one we used was `finalCombinedPlaylist.csv`.

## EDA

If you would like to explore the data, download and run `eda_lyrics_analysis.py` from the backend subrepository. Here you will see various graphs that show the distribution of the data.

## Running the Models

To run the models, download `Model_Notebook.ipynb`, and open it in Google Colab as a notebook. Make sure you also download the `finalCombinedPlaylist.csv` from the data sub repository and upload it to your Google Colab workspace. From there, you can press “Run all” and you will see our models train on the data. 

Feel free to comment out the cell that mounts the user’s Google Drive. This cell was used for plot collection purposes. If you do comment out that cell, also make sure to comment out `dir_path = os.path.join("drive", "MyDrive", "results", directory_name)` and uncomment  `# dir_path = os.path.join("results", directory_name)`.

Enjoy running the models!
