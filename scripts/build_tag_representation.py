import os
import argparse
import itertools
import networkx as nx

from tag_translation.kb.source_genres import SourceGenres
from tag_translation.kb.tag_representation import TagRepresentation
from tag_translation.utils import dump_object

"""
An important representation associated with genre taxonomies is the object TagRepresentation. It consists in
an extended and unified representation of all the source taxonomies and dbpedia, along with a parsing trie used
to tokenize genre strings. This script generates this object from the previously built taxonomies.
For more information on the TagRepresentation object, refer to `tag_translation/kb/tag_representation.py`.

"""


ope = os.path.exists
opj = os.path.join
opd = os.path.dirname


def contruct_dbpedia_dgraph(tax_dir, tag_rep):
    input_file = os.path.join(tax_dir, "dbpedia.graphml")
    source_genre = SourceGenres.from_file(input_file)

    source = source_genre.source
    genres = source_genre.genres

    graph = nx.DiGraph()

    graph.add_node(source, type=tag_rep._NODE_TYPE['SOURCE_NAME'])
    for tag in genres:
        genre = source + ":" + tag
        graph.add_node(genre, label=tag, type=tag_rep._NODE_TYPE['RAW_GENRE'])
        graph.add_edge(genre, source, type = 0)

    # Deal with adding subgenres after the genres have been added
    for tag in source_genre.genres:
        subgenres = source_genre.get_subgenres(tag)
        for subgenre in subgenres:
            graph.add_edge(source + ":" + subgenre, source + ":" + tag)

        origins = source_genre.get_origins(tag)
        for origin in origins:
            graph.add_edge(source + ":" + origin, source + ":" + tag)

        derivatives = source_genre.get_derivatives(tag)
        for deriv in derivatives:
            graph.add_edge(source + ":" + deriv, source + ":" + tag)

    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build the tag representation graph from all the taxonomies")
    parser.add_argument("taxonomy_dir", help="The directory where the taxonomies are stored")
    parser.add_argument("out_dir", help="Where to write the TagRepresentation object, the normalized graph and "
                                        "the extended graph")
    args = parser.parse_args()
    tax_dir = args.taxonomy_dir
    sources = ["dbpedia", "lastfm", "discogs", "tagtraum"]
    tag_rep = TagRepresentation()
    sg_per_source = {}
    for source in sources:
        fname = opj(tax_dir, source + ".graphml")
        assert ope(fname), f"Taxonomy for source {source} is missing. {fname} not found."
        sg_per_source[source] = SourceGenres.from_file(fname)

    dbpedia_genres = sg_per_source["dbpedia"].genres
    genres = list(itertools.chain(*[sg.genres for s, sg in sg_per_source.items() if s != "dbpedia"]))
    tag_rep.build_normalized_graph(genres, dbpedia_genres)
    tag_rep.build_extended_graph(sg_per_source.values())
    tag_rep.save_normalized_graph(opj(args.out_dir, "graph.graphml"))
    tag_rep.save_extended_graph(opj(args.out_dir, "extended_graph.graphml"))
    dump_object(tag_rep, opj(args.out_dir, "genre_rep"))
    new_db_graph = contruct_dbpedia_dgraph(tax_dir, tag_rep)
    nx.write_graphml(new_db_graph, opj(args.out_dir, "dbpedia.graphml"))
