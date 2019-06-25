import os
import sys
import csv
from tag_translation.kb.source_genres import SourceGenres


"""
Acousticbrainz provides a taxonomy associated with each source. This script converts these taxonomies into a
graphml representation, more convenient for future uses.
"""


opj = os.path.join


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: acousticbrainz_raw_repr_to_taxonomy.py acoustic_brainz_dir output_dir")
        print(" ".join(sys.argv))
        sys.exit(1)
    ab_dir = sys.argv[1]
    output_dir = sys.argv[2]
    base = opj(ab_dir, "acousticbrainz-mediaeval2017-{}-train.stats.csv.stats")
    source_files = []
    for source in ["lastfm", "discogs", "tagtraum"]:
        source_files.append(base.format(source))
    for file_path in source_files:
        print("Processing acousticbrainz ontology at {}...".format(file_path))
        source = file_path.split('-')[2]
        source_genres = SourceGenres(source)

        with open(file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter='\t')
            for row in csv_reader:
                genre_tags = row['genre/subgenre'].split("---")

                if len(genre_tags) ==  1:
                    # Add the main genre, no subgenre given
                    source_genres.add_genre(genre_tags[0])
                else:
                    # Add the genre subgenre information
                    source_genres.add_subgenre(genre_tags[0], genre_tags[1])
        print("{} genres in the source_genres repr".format(len(source_genres.genres)))
        out_name = opj(output_dir, source_genres.source) + ".graphml"
        print("Saving representation at {}".format(out_name))
        source_genres.save_graphml(out_name)
