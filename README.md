# MusicGenreTranslation

Python code for reproducing music genre translation experiments presented in the paper *Leveraging knowledge bases and parallel annotations for music genre translation* at ISMIR 2019.


# Organisation

- The directory "scripts" contains standalone python scripts.
- The directory "tag_translation" contains the python module used to perform
tag translation.
- The directory "docker" contains the Dockerfile that defines the image in
which the experiment's code can be run.

# Simply reproduce the results

Make sure you have Docker installed, and run 

```
git clone git@github.com:deezer/MusicGenreTranslation.git
cd MusicGenreTranslation
./build_and_run.sh
``` 

This will:

- build a docker image called `tag_translation_research` in which the
experiments will be run.
- launch a container where all the scripts to produce the articles' figures
will be run. 
- download all the groundtruth files for train and validation data in the common recordings
dataset, along with the taxonomies provided by acousticbrainz for the 
3 sources lastfm, discogs and tagtraum. Additionally, a mapping between recodingmbids and
artistmbids will also be downloaded. The artist information is need to perform the
stratified sampling when generating folds.
- run all the necessary scripts to generate the final plots

After this, if everything went well, the figures will be in `data/plots`.

/!\ Because running all the results is quite long, the plots are generated only
for the lowest amounts of data, but you can change that setting in
`tag_translation/conf.py`.

# Additional notes

To have better control of how the scripts are executed as well as the
environment you can refer to `tag_translation/conf.py`.
