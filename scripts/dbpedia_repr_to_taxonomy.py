import sys
import os
from SPARQLWrapper import SPARQLWrapper, JSON

from tag_translation.kb.source_genres import SourceGenres


"""
This script is used to build the DBpedia taxonomy from resources tagged with dbo:MusicGenre. This is simple and
naive approach that leads to a very noisy taxonomy, but which is useful to design a baseline.
"""


opj = os.path.join

def get_genre_name_from_dbpedia_resource(genre):
    """
        Get the name of a resource in dbpedia, in this case the genre
    """
    genre = genre.replace('http://dbpedia.org/resource/', '')
    return genre


def get_dbpedia_music_genres():
    """
        Get the genres found in dbpedia
    """
    global sparql_dbpedia
    sparql_dbpedia.setQuery("""
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT distinct ?genre
        WHERE {
            ?genre rdf:type ?o.
            FILTER (?o IN (dbo:MusicGenre))
        }
        """
    )

    sparql_dbpedia.setReturnFormat(JSON)
    results = sparql_dbpedia.query().convert()

    genres = set()
    for result in results["results"]["bindings"]:
        genres.add(get_genre_name_from_dbpedia_resource(result["genre"]["value"]))
    return genres


def get_dbpedia_music_genre_derivatives():
    """
        Get the derivatives of genres found in dbpedia
    """
    sparql_dbpedia.setQuery("""
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT distinct ?genre, ?derivative
        WHERE {
            ?genre rdf:type dbo:MusicGenre .
            ?derivative dbo:derivative ?genre
        }
        """
    )

    sparql_dbpedia.setReturnFormat(JSON)
    results = sparql_dbpedia.query().convert()

    genre_derivatives = {}
    for result in results["results"]["bindings"]:
        key = get_genre_name_from_dbpedia_resource(result["genre"]["value"])
        if key not in genre_derivatives:
            genre_derivatives[key] = []
        value = get_genre_name_from_dbpedia_resource(result["derivative"]["value"])
        genre_derivatives[key].append(value)

    return genre_derivatives


def get_dbpedia_music_genre_origins():
    """
        For each genre it shows the genres that originated from this one
    """
    sparql_dbpedia.setQuery("""
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT distinct ?genre, ?origin
        WHERE {
            ?genre rdf:type dbo:MusicGenre .
            ?origin dbo:stylisticOrigin ?genre
        }
        """
    )

    sparql_dbpedia.setReturnFormat(JSON)
    results = sparql_dbpedia.query().convert()

    genre_origins = {}
    for result in results["results"]["bindings"]:
        key = get_genre_name_from_dbpedia_resource(result["genre"]["value"])
        if key not in genre_origins:
            genre_origins[key] = []
        value = get_genre_name_from_dbpedia_resource(result["origin"]["value"])
        genre_origins[key].append(value)

    return genre_origins


def get_dbpedia_music_genre_subgenres():
    """
        Get the subgenres of genres found in dbpedia
    """
    sparql_dbpedia.setQuery("""
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT distinct ?genre, ?subgenre
        WHERE {
            ?genre rdf:type dbo:MusicGenre .
            ?subgenre dbo:musicSubgenre ?genre
        }
        """
    )

    sparql_dbpedia.setReturnFormat(JSON)
    results = sparql_dbpedia.query().convert()

    genre_subgenres = {}
    for result in results["results"]["bindings"]:
        key = get_genre_name_from_dbpedia_resource(result["genre"]["value"])
        if key not in genre_subgenres:
            genre_subgenres[key] = []
        value = get_genre_name_from_dbpedia_resource(result["subgenre"]["value"])
        genre_subgenres[key].append(value)

    return genre_subgenres


def get_dbpedia_music_genre_wikiredirects():
    """
        Get the aliases of genres found in dbpedia, as specified in the wiki redirects
    """
    sparql_dbpedia.setQuery("""
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT distinct ?genre, ?wikir
        WHERE {
            ?genre rdf:type dbo:MusicGenre .
            ?wikir dbo:wikiPageRedirects ?genre
        }
        """
    )

    sparql_dbpedia.setReturnFormat(JSON)
    results = sparql_dbpedia.query().convert()

    wiki_redirects = {}
    for result in results["results"]["bindings"]:
        key = get_genre_name_from_dbpedia_resource(result["genre"]["value"])
        if key not in wiki_redirects:
            wiki_redirects[key] = []
        value = get_genre_name_from_dbpedia_resource(result["wikir"]["value"])
        wiki_redirects[key].append(value)

    return wiki_redirects


if __name__ == "__main__":
    """
        Process the genre tags and relations from Dbpedia
    """

    if len(sys.argv) != 2:
        print("Usage: dbpedia_repr_to_taxonomy.py output_dir")
        sys.exit(1)
    output_dir = sys.argv[1]

    sparql_dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")

    source_genres = SourceGenres('dbpedia')

    genres = get_dbpedia_music_genres()
    for genre in genres:
        source_genres.add_genre(genre)

    genre_derivatives = get_dbpedia_music_genre_derivatives()
    for genre in genre_derivatives:
        for derivative in genre_derivatives[genre]:
            source_genres.add_derivative(genre, derivative)

    genre_origins = get_dbpedia_music_genre_origins()
    for genre in genre_origins:
        for origin in genre_origins[genre]:
            source_genres.add_stylistic_origin(genre, origin)

    genre_supragenres = get_dbpedia_music_genre_subgenres()
    for genre in genre_supragenres:
        for supragenre in genre_supragenres[genre]:
            source_genres.add_subgenre(supragenre, genre)

    wiki_redirects = get_dbpedia_music_genre_wikiredirects()
    for genre in wiki_redirects:
        for alias in wiki_redirects[genre]:
            source_genres.add_alias(genre, alias)
        # In dbpedia genres come also separated by _, add both this form plus replacing the _ by space
        source_genres.add_alias(genre, genre.replace('_', ' '))

    print("{} genres in the source_genres repr".format(len(source_genres.genres)))
    out_name = opj(output_dir, source_genres.source) + ".graphml"
    print("Saving representation at {}".format(out_name))
    source_genres.save_graphml(out_name)