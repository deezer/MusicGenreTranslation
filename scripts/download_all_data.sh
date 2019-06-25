#!/usr/bin/env bash


download_file(){
    drive_id=$1
    dest=$2

    if [ ! -e ${dest%.bz2} ]; then
        wget https://drive.google.com/uc\?export\=download\&id\="$drive_id" -O $dest
    else
        echo "$dest already exists. Skipping."
    fi
}


extract_all(){
    for f in $@; do
        if [ ! -e ${f%.bz2} ]; then
            bzip2 -d $f
        fi
    done
}


download_all(){
    download_file "1gRrcSxv5y8XppUdaTf39Xy-gmcAPEq_n" ab_mediaeval_val_discogs.tsv.bz2
    download_file "1p16FRaJPzPSQrzOYI8BIP2LyFs2KpQZb" ab_mediaeval_val_lastfm.tsv.bz2
    download_file "15fP1hy-uL5WqYZ4QY9ImdanNYukaMkT2" ab_mediaeval_val_tagtraum.tsv.bz2
    download_file "0B8wz5KkuLnI3NkczWHRiX2FjbmM" ab_mediaeval_train_discogs.tsv.bz2
    download_file "0B8wz5KkuLnI3RWVZaVhEY09ETWc" ab_mediaeval_train_lastfm.tsv.bz2
    download_file "0B8wz5KkuLnI3aXY5c0NqY25HQ2c" ab_mediaeval_train_tagtraum.tsv.bz2
    download_file "0B9efYsv7Y7gpZUFSYjlJaXJhbVk" acousticbrainz-mediaeval2017-discogs-train.stats.csv.stats
    download_file "0B9efYsv7Y7gpRGh6NEdIMVJ3Rk0" acousticbrainz-mediaeval2017-lastfm-train.stats.csv.stats
    download_file "0B9efYsv7Y7gpSTZyeXlQREhsOWc" acousticbrainz-mediaeval2017-tagtraum-train.stats.csv.stats
    download_file "1W04C4bMwiNuMgggIoODl--6dVBs_vWxE" recordingmbid2artistmbid.tsv

    extract_all ab_mediaeval_val_discogs.tsv.bz2 ab_mediaeval_val_lastfm.tsv.bz2 ab_mediaeval_val_tagtraum.tsv.bz2 \
                ab_mediaeval_train_discogs.tsv.bz2 ab_mediaeval_train_lastfm.tsv.bz2 ab_mediaeval_train_tagtraum.tsv.bz2
}


# Convert the path to absolute path
PATH_SCRIPT="`dirname \"$0\"`"
PATH_SCRIPT="`( cd \"$PATH_SCRIPT\" && pwd )`"
if [ -z "$PATH_SCRIPT" ] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1
fi
echo "$PATH_SCRIPT"

DATA_DIR=$(dirname $PATH_SCRIPT)/data
AB_DIR=$DATA_DIR/acousticbrainz

echo $AB_DIR
mkdir -p $AB_DIR

cd $AB_DIR && download_all


