import os
import pandas as pd
import pickle
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from tag_translation.kb.dbpedia_mapping import map_taxonomy_on_dbpedia, map_taxonomies_direct
from tag_translation.kb.source_genres import SourceGenres

"""
This script builds a translation table between source and target taxonomies. This is a matrix that assigns a
translation score between each pair of source and target tag. This is a pure Knowledge-Based approach and the scores
are based on some heuristics depending on how the tags were matched together.

"""

def get_kb_translation_dbpdis(sources, target, tag_rep, dbp_graph, tax_dir):
    """
        Get translation table with dbpedia disambiguation
    """

    result_tbl = None
    target_source_genre = SourceGenres.from_file(os.path.join(tax_dir, target + ".graphml"))
    target_tbl = map_taxonomy_on_dbpedia(target_source_genre, tag_rep, dbp_graph).transpose()
    target_genres = tag_rep.get_normalized_genres_for_source(target)
    inv_target_genres_table = {}
    for norm, corresp in target_genres.items():
        for genre in corresp:
            inv_target_genres_table[genre] = norm

    for source in sources:
        print(f"Mapping for source {source}")
        # create translation table
        source_genres = tag_rep.get_normalized_genres_for_source(source)
        inv_source_genres_table = {}
        for norm, corresp in source_genres.items():
            for genre in corresp:
                inv_source_genres_table[genre] = norm
        source_genre = SourceGenres.from_file(os.path.join(tax_dir, source + ".graphml"))
        source_tbl = map_taxonomy_on_dbpedia(source_genre, tag_rep, dbp_graph).transpose()

        final_tbl = pd.concat([source_tbl, target_tbl], join="outer")
        final_tbl = final_tbl.groupby(final_tbl.index).agg('mean')

        dist = cosine_similarity(final_tbl)

        translation_tbl = pd.DataFrame(dist, index=final_tbl.index, columns=final_tbl.index)
        translation_tbl = translation_tbl.loc[source_genres.keys(), target_genres.keys()]

        dir_trans_tbl = map_taxonomies_direct(source_genre, target_source_genre, tag_rep)
        translation_tbl = (translation_tbl + dir_trans_tbl) / 2

        translation_tbl.index.name = 'source'

        result = pd.DataFrame(0, index=inv_source_genres_table.keys(), columns=inv_target_genres_table.keys())
        for sg in inv_source_genres_table:
            norm_sg = inv_source_genres_table[sg]
            for tg in inv_target_genres_table:
                norm_tg = inv_target_genres_table[tg]
                # print(norm_sg, norm_tg)
                result.loc[sg, tg] = translation_tbl.loc[norm_sg, norm_tg]
        if result_tbl is None:
            result_tbl = result
        else:
            result_tbl = result_tbl.merge(
                result, on=list(result_tbl.columns.values), how="outer", left_index=True, right_index=True)

    result_tbl = result_tbl.groupby(result_tbl.index).agg('mean')
    return result_tbl


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sources", nargs='+', required=True)
    parser.add_argument("-t", "--target", required=True)
    parser.add_argument("--tag-rep", required=True, help="Path of the pickle file containing the tag representation.")
    parser.add_argument("--tax-dir", required=True, help="Path to the directory containing the taxonomies for all "
                                                         "sources, target, and dbpedia")
    parser.add_argument("-o", "--out", required=True, help="Where to write the translation table in csv format")
    args = parser.parse_args()

    tag_rep_path = os.path.join(args.tag_rep, "genre_rep")
    db_grap_path = os.path.join(args.tag_rep, "dbpedia.graphml")
    with open(tag_rep_path, 'rb') as pickle_file:
        tag_rep = pickle.load(pickle_file)
    db_graph = nx.read_graphml(db_grap_path)

    table = get_kb_translation_dbpdis(args.sources, args.target, tag_rep, db_graph, args.tax_dir)
    table.to_csv(args.out)
