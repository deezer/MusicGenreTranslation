import numpy as np
import pandas as pd
import networkx as nx

from tag_translation.kb.tag_representation import TagRepresentation


def get_source_supragenres(source_genre_obj, sg, source_genres):
    """
        Retrieve the origin of a genre
    """
    supragenres = set()
    for orig_sg in source_genres[sg]:
        orig_sg = orig_sg.replace(source_genre_obj.source + ":", "")
        for g in source_genre_obj.genres:
            l = source_genre_obj.get_subgenres(g)
            if l is not None and orig_sg in l:
                supragenres.add(g)
    return supragenres


def compute_results_from_centrality_scores(mappings, dbp_graph, factor=1.):
    result = {}
    if len(mappings) != 0:
        # If it was not an exact match but candidates were found return the best candidates
        ord_cand_scores = get_candidates_ordered_by_centrality(mappings, dbp_graph)
        best_candidates = sorted([g for (g,v) in ord_cand_scores if v == ord_cand_scores[0][1]])
        confidence = [1 / len(best_candidates) * factor for _ in range(len(best_candidates))]
        result.update({g: c for g, c in zip(best_candidates, confidence)})
    return result


def get_candidates_ordered_by_centrality(candidates, graph):
    """
        Return candidates ordered by their in-degree centrality in the subgraph composed only of those
    """
    # Extract subgraph from the main graph containing only the candidates
    for c in candidates:
        assert c in graph, "candidate {} is not in graph".format(c)
    subgraph = graph.subgraph(candidates)
    # Compute the indegree centrality scores
    centra_scores = nx.in_degree_centrality(subgraph)
    # Sort genres by the centrality scores
    sorted_dbp_genres_genres = sorted(centra_scores, key=centra_scores.get, reverse=True)
    return [(genre, centra_scores[genre]) for genre in sorted_dbp_genres_genres]


def centrality_mapping_main_genres(sg, tag_rep):
    """
        The strategy to find it is based on the following hypothesis:
        The main genres will have a high centrality degree in the dbpedia graph
    """
    # Find out all the dbpedia genres which have the normalized form containing the genre/concept sg
    dbp_genres = set()
    for path in nx.all_simple_paths(tag_rep._ext_graph, source=sg, target="dbpedia", cutoff=3):
        dbp_genres.add(path[2])

    # Remove the aliases from this set because they are not relevant for establishing the centrality
    return tag_rep.filter_out_aliases(dbp_genres)


def centrality_mapping_composed_genres(sg, tag_rep):
    """
        Find mappings for the genres composed and not found exactly in Dbpedia e.g. british invasion
        Return the candidates and an info knowing if the genre is part of another genre (all concepts are contained
        in the other genre plus some more)
    """
    # Step 1: enumerate the paths from the normalized genres sg to "dbpedia" of maximum length 4
    # this will identify Dbpedia genres that share at least a word with sg
    # Also keep track of how many concepts are shared
    dbp_genres = {}
    ext_graph = tag_rep._ext_graph
    for path in nx.all_simple_paths(ext_graph, source=sg, target="dbpedia", cutoff=4):
        index_g = 2 if len(path) == 4 or ext_graph.node[path[2]]['type'] == 4 else 3
        dbp_g = path[index_g]

        if dbp_g not in dbp_genres:
            dbp_genres[dbp_g] = set()
        # Path[1] is a normalized string that sg shares with another dbpedia genre
        dbp_genres[dbp_g].add(path[1])

    # Compose the candidates list by selecting all the genres which share all concepts of sg with sg
    candidates = set()
    for dbp_g in dbp_genres:
        if len(' '.join(dbp_genres[dbp_g])) == len(sg):
            candidates.add(dbp_g)

    if len(candidates) > 0:
        candidates = tag_rep.filter_out_aliases(candidates)
        return candidates, True

    if len(dbp_genres) > 0:
        # find out what is the maximum number of concepts shared by a genre
        max_no_concepts = max([len(dbp_genres[dbp_g]) for dbp_g in dbp_genres])
        # extract as candidates the genres that contain the maximum number of concepts
        candidates = [dbp_g for dbp_g in dbp_genres if len(dbp_genres[dbp_g]) == max_no_concepts]
        # if the maximum number of concepts do not contain a main genre than exclude those
        for dbp_g in dbp_genres:
            if len(dbp_genres[dbp_g]) == max_no_concepts:
                one_main_genre = False
                for concept in dbp_genres[dbp_g]:
                    if ext_graph.node[concept]['type'] == 3:
                        one_main_genre = True
                        break
                if not one_main_genre:
                    candidates.remove(dbp_g)
        # if through the previous filtering no candidates are found take all the genres
        if len(candidates) == 0:
            candidates = set(dbp_genres.keys())

        candidates = tag_rep.filter_out_aliases(candidates)
    return candidates, False


def map_taxonomies_direct(source_genre_obj, target_genre_obj, tag_rep, norm=True):
    if not norm:
        source_genres = source_genre_obj.genres
        target_genres = target_genre_obj.genres
    else:
        source_genres = tag_rep.get_normalized_genres_for_source(source_genre_obj.source)
        target_genres = tag_rep.get_normalized_genres_for_source(target_genre_obj.source)

    translation_tbl = pd.DataFrame(np.zeros(
        (len(source_genres), len(target_genres))), index=source_genres, columns=target_genres)
    for sg in source_genres:
        if sg in target_genres:
            translation_tbl.loc[sg, sg] = 1

    return translation_tbl


def get_dbpedia_mappings(sg, source_genres, dbpedia_genres, tag_rep, dbp_graph, source_genre_obj):

    print("SG IS {}".format(sg))
    # Case 1: the genre is directly found among the dbpedia genres
    result = {}
    ext_graph = tag_rep._ext_graph
    if sg in dbpedia_genres:
        candidates = tag_rep.filter_out_aliases(dbpedia_genres[sg])
        confidence = [1 / len(candidates)]*len(candidates)
        result = {g: c for g, c in zip(candidates, confidence)}
        return result, "case 1"

    # Case 2: the supra-genres can help to disambiguate sg,
    # do it only if it is not a main genre

    if not source_genre_obj.is_root(sg):
        supragenres = get_source_supragenres(source_genre_obj, sg, source_genres)
        # print("Supragenres are {}".format(supragenres))
        not_mapped = set()
        candidates = set()
        for g in supragenres:
            # The idea is to try to see if the decoded genre written as "origin + genre" is found instead in Dbpedia
            g = g.replace(source_genre_obj.source + ":", "")
            ext_genre_name = tag_rep.decode_genre(g + " " + sg)
            if ext_genre_name in dbpedia_genres:
                candidates.update(tag_rep.filter_out_aliases(dbpedia_genres[ext_genre_name]))
            else:
                # print("{} not in dbpedia genres".format(ext_genre_name))
                not_mapped.add(ext_genre_name)

        if len(candidates) > 0 and len(not_mapped) == 0:
            confidence = [1 / len(candidates)] * len(candidates)
            result = {g: c for g, c in zip(candidates, confidence)}
            return result, "case 2"

    # Case 3 map the rest using the graph centrality
    # if there are still genres in not_mapped then try the centrality strategy applied for the
    # main genre
    assert sg in ext_graph, "genre {} from source {} is not in ext_graph".format(sg, source_genre_obj.source)

    if ext_graph.node[sg]['type'] == TagRepresentation._NODE_TYPE["GENRE_AND_CONCEPT"]:
        # 3.1: the genre is a main genre, it should be found in the dbpedia, but probably under another name
        mappings = centrality_mapping_main_genres(sg, tag_rep)
        result.update(compute_results_from_centrality_scores(mappings, dbp_graph, factor=1.))
        case = "case 3.1"
    elif ext_graph.node[sg]['type'] == TagRepresentation._NODE_TYPE["GENRE"]:
        # 3.2: the genre is a composed genre
        mappings, high_confidence = centrality_mapping_composed_genres(sg, tag_rep)
        if high_confidence:
            result.update(compute_results_from_centrality_scores(mappings, dbp_graph, factor=.1))
        else:
            result.update(compute_results_from_centrality_scores(mappings, dbp_graph, factor=.1))
        case = "case 3.2"
    else:
        case = "case 3.3"
        print("WARNING unexpected type {}".format(ext_graph.node[sg]['type']))
    return result, case


def map_taxonomy_on_dbpedia(source_genre, tag_rep, dbp_graph):
    source = source_genre.source

    # step 1: Normalize all genres
    source_genres = tag_rep.get_normalized_genres_for_source(source_genre.source)
    dbpedia_genres = tag_rep.get_normalized_genres_for_source("dbpedia")
    dpb_mapping = pd.DataFrame(np.zeros(
        (len(dbpedia_genres), len(source_genres))), index=dbpedia_genres, columns=source_genres)

    not_mapped = set()
    empty = 0
    for sg in source_genres:
        # step 2: Find neighbours of sg in the extended graph taxonomy

        likely_genres, case = get_dbpedia_mappings(
            sg, source_genres, dbpedia_genres, tag_rep, dbp_graph, source_genre)
        if len(likely_genres) == 0:
            empty += 1
        if likely_genres is not None:
            for g in likely_genres:
                g_norm = tag_rep.decode_genre(g.replace("dbpedia:", ""))
                dpb_mapping.loc[g_norm, sg] = likely_genres[g]

                dbp_candidates = tag_rep.get_dbpedia_neighbours_by_relation(g, norm=False)
                for rel in dbp_candidates:
                    if rel not in [2, 3]:
                        continue

                    for dbp_c in dbp_candidates[rel]:
                        dbp_c_norm = tag_rep.decode_genre(dbp_c.replace("dbpedia:", ""))
                        dpb_mapping.loc[dbp_c_norm, sg] = likely_genres[g] * .5
        else:
            not_mapped.add(sg)

    # For the not mapped genres copy the information its supragenres and subgenres
    for sg in not_mapped:
        related_genres = set()
        for orig_sg in source_genres[sg]:
            orig_sg = orig_sg.replace(source + ":", "")
            l = source_genre.get_subgenres(orig_sg)
            if l is not None:
                related_genres.update(l)

            for g in source_genre.genres:
                l = source_genre.get_subgenres(g)
                if l is not None and orig_sg in l:
                    related_genres.add(g)

        aprox_genres = [tag_rep.decode_genre(g) for g in related_genres]
        dpb_mapping[sg] = dpb_mapping[aprox_genres].mean(axis=1) * 1.

    return dpb_mapping
