import logging
import re
import networkx as nx
from unidecode import unidecode
from alphabet_detector import AlphabetDetector

from tag_translation.kb.trie import Trie
from tag_translation.kb.source_genres import SourceGenres

logger = logging.getLogger(__name__)


class TagRepresentation(object):
    """
        It derives a representation of genres from multiple sources
        It is based on multiple elements:
            - a trie to efficiently store the varied tokens related to genre
                the trie also helps to tokenize the concepts of genres written without space
            - a graph to represent relations between tokens
            - a dictionary of aliases
    """

    # Regex to identify short strings containing & (or similar char) such as r&b, d+b
    # Here & is part of the word
    _REG_AND_PARTW = re.compile(r"(?P<prefix>\b[\w'!.’])(?P<and>\s*([&+-])\s*)(?P<suffix>[\w'!.’]\b)")

    _REG_AND_PARTWII = re.compile(r"(?P<prefix>\b[\w'!.’])(?P<and>\s*[&+-]\s*)(?P<suffix>[\w'!.’]*\b)")

    # Regex to identify longer words which are linked through & (or similar char) such as "rock & roll"
    # Here & has the role of conjunction
    _REG_AND_CONJ = re.compile(r"\b((?P<prefix>[\w'!.’]*)\s*[&+-])?\s*(?P<suffix>[\w'!.’]+)*$")

    # Regex used to tokenize by any non-alphabet letter except ’'!.
    _REG_TOKENIZE = re.compile(r"[\w’'!.]+")

    # Regex used to normalize strings containing any of ’'!.
    _REG_NORMALIZE = re.compile(r"[’'!.]")

    _ALPHABET_DETECT = AlphabetDetector()

    _NODE_TYPE = {"GENRE": 1, "CONCEPT": 2, "GENRE_AND_CONCEPT": 3, "RAW_GENRE": 4, "SOURCE_NAME": 5}

    _EDGE_TYPE = {"ALIAS": 1, "SUBGENRE": 2, "ORIGIN": 3, "DERIVATIVE": 4}

    def __init__(self):
        self._trie = Trie()
        self._aliases = {}

        self._tokens = set()
        self._edges = {}
        self._normalized_graph = None
        self._ext_graph = None

        self._tags_otherlanguages = set()

    def normalize_token(self, token, update_self):
        """
            Normalize a token by replacing non-ascii and by unidecoding
        """
        normalized = self._REG_NORMALIZE.sub('', token)

        # If the unidecoded version contains non-ascii then save the token for being dealt with later
        # These tokens come from different languages such as Chinese, Russian
        if not self._ALPHABET_DETECT.is_latin(normalized):
            if update_self:
                self._tags_otherlanguages.add(token)
            return None

        normalized = unidecode(normalized)
        if not normalized.isalnum():
            return None

        return normalized

    def _basic_str_tokenization(self, tag, update_self):
        """
           Split string in words by non-chars
        """

        # First transform the string to lower case as a prerequisite
        # Also manually replace _
        string = tag.lower().replace('_', ' ')

        # Replace & and such chars as part of the word with n (see _REG_AND_PARTW)
        # Do it manually (instead sub) because that specific token must be added to aliases

        match = self._REG_AND_PARTW.search(string)
        if match is not None:
            d = match.groupdict()
            before = str(string)
            string = string.replace(d['and'], 'n')
            if update_self:
                self._add_to_aliases(string, before)

        string = self._REG_AND_PARTWII.sub(r'\g<prefix> \g<suffix>', string)

        # Replace & and such chars as conjunctions with space (see _REG_AND_CONJ)
        string = self._REG_AND_CONJ.sub(r'\g<prefix> \g<suffix>', string)

        tokens = self._REG_TOKENIZE.findall(string)

        return set(tokens)

    def _process_and_update(self, tag):
        """
            Process each tag (tokenization, normalization)
            Initializes the internal structures too: the aliases dict and graph
        """
        if len(tag) == 1:
            return

        tag = tag.lower()

        # Tokenize the tag
        tag_tokens = self._basic_str_tokenization(tag, update_self=True)

        for token in tag_tokens:
            # Normalize each token
            normalized = self.normalize_token(token, update_self=True)
            if normalized is None:
                # This situation happens if the tag is in another language with special alphabets
                # Ignore these tags for now, will be treated as future work
                continue

            # If the normalized is different than the initial token
            # Save the initial token as an alias
            if normalized != token:
                self._add_to_aliases(normalized, token)

            # Save the normalized token for being added to the trie
            self._tokens.add(normalized)

            # Save the edge between the tag and the tokens
            # It will be added to the graph after the trie creation
            if normalized not in self._edges:
                self._edges[normalized] = set()
            self._edges[normalized].add(tag)

    def _add_to_aliases(self, key, value):
        """
            Add a key, value to the aliases dictionary
        """
        if key not in self._aliases:
            self._aliases[key] = set()
        self._aliases[key].add(value)

    def build_normalized_graph(self, genre_tags, dbpedia_genres=[]):
        """
            Build representation for genres in several steps:
            - extract tokens through basic string tokenization
            - adds the tokens to the trie
            - update the graph nodes and edges
            - update the aliases dictionary
        """
        self._normalized_graph = nx.Graph()
        for tag in dbpedia_genres:
            self._process_and_update(tag)
        dbpedia_tokens = set(self._tokens)

        for tag in genre_tags:
            self._process_and_update(tag)
        tokens = self._tokens

        sorted_tokens = sorted(tokens, key=len)
        for token in sorted_tokens:
            trie_concepts = [token]
            # If it is in dbpedia then it has been already added
            if token in dbpedia_tokens:
                if len(token) < 7:
                    self._trie._add_word(token)
                else:
                    trie_concepts = self._trie.tokenize(token)
                    if len(trie_concepts) == 0 or any([len(c) <= 2 for c in trie_concepts]):
                        trie_concepts = [token]
                        self._trie._add_word(token)
            else:
                trie_concepts = self._trie.add_string_with_tokenization(token)

            processed_token = " ".join(trie_concepts)

            # This is the situation of token = latindance, trie_concepts = [latin, dance]
            if len(trie_concepts) > 1:
                # Update all the tags in edges containing this token
                self._edges[token].update(set([tag.replace(token, processed_token) for tag in self._edges[token]]))

            # In self._edges are always genres
            for tag in self._edges[token]:
                if not self._normalized_graph.has_node(tag):
                    # If the tag does not exist in the graph add it
                    self._normalized_graph.add_node(tag)
                    # and mark it as genre
                    self._normalized_graph.node[tag]['type'] = self._NODE_TYPE["GENRE"]

                elif self._normalized_graph.node[tag]['type'] == self._NODE_TYPE["CONCEPT"]:
                    # if the tag existed in the graph but as concept,
                    # update its type as genre and concept
                    self._normalized_graph.node[tag]['type'] = self._NODE_TYPE["GENRE_AND_CONCEPT"]

                for concept in trie_concepts:
                    if tag != concept:
                        # if the concept did not exist, add it
                        if not self._normalized_graph.has_node(concept):
                            self._normalized_graph.add_node(concept)
                            self._normalized_graph.node[concept]['type'] = self._NODE_TYPE["CONCEPT"]

                        # if it existed, it means it was added as genre, so update its type
                        elif self._normalized_graph.node[concept]['type'] == self._NODE_TYPE["GENRE"]:
                            self._normalized_graph.node[concept]['type'] = self._NODE_TYPE["GENRE_AND_CONCEPT"]
                        self._normalized_graph.add_edge(tag, concept, type = 0)

        self.cluster_nodes_sharing_same_concepts()
        self._normalized_graph.remove_edges_from(self._normalized_graph.selfloop_edges())

    def decode_genre(self, genre):
        """
            Given a genre it decodes it by first applying tokenization, and after using the trie's decoding procedure
        """
        norm_tokens = set()

        # First tokenize genre using regex
        tag_tokens = self._basic_str_tokenization(genre, update_self=False)

        for token in tag_tokens:
            # Second, normalize each token
            normalized = self.normalize_token(token, update_self=False)
            if normalized is None:
                # This situation happens if the tag is in another language with special alphabets
                # Ignore these tags for now, will be treated as future work
                break

            # Third tokenize genre using trie
            result = self._trie.decode_tag(normalized)

            if len(result) == 0:
                #print("!!! decoded genre unknown", genre)
                logger.warning(f"Unknown decoded genre {genre}")
                # Genre is not known
                break
            norm_tokens.update(result)

        # Finally join the sorted tokens
        decoded = " ".join(sorted(norm_tokens))

        return decoded

    def print_concepts(self, ordered_by_len=False):
        print ("Printing the trie concepts")
        concepts = self._trie.get_all_words()
        if ordered_by_len:
            concepts.sort(key = len)
        for concept in concepts:
            print(concept)

    def print_aliases(self):
        for concept in self._aliases:
            print(concept, self._aliases[concept])

    def print_concept_trie_tokenization(self, min_no_tokens=1):
        print("\n Trie tokenization of the existing tokens is: ")
        for token in self._tokens:
            concepts = self._trie.decode_tag(token)
            if len(concepts) >= min_no_tokens:
                print(token, " >> ", concepts)

    def print_graph(self):
        print("\n Graph nodes are")
        print(self._normalized_graph.nodes())

        print("\n Graph edges are ")
        print(self._normalized_graph.edges())

    def save_normalized_graph(self, file_name):
        assert self._normalized_graph is not None
        nx.write_graphml(self._normalized_graph, file_name)

    def save_extended_graph(self, file_name):
        assert self._ext_graph is not None
        nx.write_graphml(self._ext_graph, file_name)

    def cluster_nodes_sharing_same_concepts(self):
        """
            Save a new graph where nodes that mean the same thing
            such as "jazz-rock" "rock jazz" are clustered in a single node
            Each node contain a set of aliases
        """
        ngraph = nx.Graph()

        # Add first the concepts and genre-concepts
        for label in self._normalized_graph.nodes():
            if self._normalized_graph.node[label]['type'] != self._NODE_TYPE["GENRE"]:
                ngraph.add_node(label, **self._normalized_graph.node[label])

        # Then deal with the composed genres
        for label in self._normalized_graph.nodes():
            if self._normalized_graph.node[label]['type'] == self._NODE_TYPE["GENRE"]:
                clean_genre = label
                neighbours = list(self._normalized_graph.neighbors(label))
                if len(neighbours) > 0:
                    # This is when a genre tag is composed of multiple concepts / genre-concepts
                    # Create a normal form, by concatenating the sorted labels
                    clean_genre = " ".join(sorted(neighbours))

                if not ngraph.has_node(clean_genre):
                    ngraph.add_node(clean_genre, **self._normalized_graph.node[label])

                for n in neighbours:
                    ngraph.add_edge(clean_genre, n, type = 0)

        self._normalized_graph = ngraph

    def check_prefix(self, max_cost = 1, min_length = 4):
        """
            Groups concepts together by using the trie to find similar concepts
            (2 concepts are similar if they share the same prefix)

            max_cost is the maximum length difference between strings
            min_length is the minimum length of a string to be considered for grouping (if the strings are too short then one could expect much more noise)
        """
        print("\nTrie similarity \n")
        concepts = self._trie.get_all_words()
        concepts.sort(key=len)

        groups = []
        while len(concepts) > 0:
            concept = concepts.pop(0)
            if len(concept) < min_length:
                continue

            # Search strings that start with prefix concept
            words = self._trie.get_words(concept)
            results = [word for word in words if abs(len(concept) - len(word)) <= max_cost]
            if len(results) > 1:
                groups.append(results)
                print(groups[-1])

            # Delete the concepts that have been already visited
            for result in results:
                if result in concepts:
                    concepts.remove(result)
        return groups

    def check_lemmas(self, stemmer):
        """
            Groups words together by using stemming
            default stemmer is Lancaster
        """
        if stemmer is None:
            return []

        print("\nSimilarity by stemming with " + type(stemmer).__name__ + "\n")

        concepts = self._trie.get_all_words()
        results = {}

        for concept in concepts:
            lemma = stemmer.stem(concept)

            if lemma not in results:
                results[lemma] = []
            results[lemma].append(concept)

        groups = []
        for lemma in results:
            if len(results[lemma]) > 1:
                groups.append(results[lemma])
                print(groups[-1])

        return groups

    def get_normalized_genres_for_source(self, source):
        """
            Return a dictionary with normalized genres as keys and the original genres as values
        """
        assert self._ext_graph is not None
        result = {}
        for n in self._ext_graph.neighbors(source):
            if n == source or not n.startswith(source):
                continue

            # Normalize genre
            norm_g = self.decode_genre(n.replace(source + ":", ""))
            if norm_g not in result:
                result[norm_g] = []
            result[norm_g].append(n)
        return result

    def build_extended_graph(self, source_genres):
        """
        Creates the extended graph from a list of SourceGenres objects. The extended graph connects the source
        taxonomies to the normalized graph.

        :param source_genres: List of SourceGenres objects.
        :return:
        """
        assert self._normalized_graph is not None, "You must build the normalized graph before building " \
                                                   "the extended graph"
        self._ext_graph = self._normalized_graph.copy()
        for sg in source_genres:
            source = sg.source
            genres = sg.genres

            # print("\n**********  ", source, " *************")
            self._ext_graph.add_node(source, type=self._NODE_TYPE['SOURCE_NAME'])
            for tag in genres:
                genre = source + ":" + tag
                self._ext_graph.add_node(genre, label=tag, type=self._NODE_TYPE['RAW_GENRE'])

                # add edge between source and genre
                self._ext_graph.add_edge(source, genre, type=0)

                decoded = self.decode_genre(tag)
                if decoded in self._ext_graph:
                    # If the genre can be decoded add a edge to the decoded genre
                    self._ext_graph.add_edge(genre, decoded, type=0)
                    # print(tag, " ---> ", decoded)
                else:
                    print(tag, "UNKNOWN, won't be added to the extended graph")

            # Deal with adding subgenres after the genres have been added
            self.add_taxonomy_relations_to_extended_graph(sg)
        self._ext_graph.remove_edges_from(self._ext_graph.selfloop_edges())

    def add_taxonomy_relations_to_extended_graph(self, source_genre):
        sg_rtype2_tr_etype = {SourceGenres.SUBGENRES: self._EDGE_TYPE["SUBGENRE"],
                              SourceGenres.STORIGINS: self._EDGE_TYPE["ORIGIN"],
                              SourceGenres.ALIASES: self._EDGE_TYPE["ALIAS"],
                              SourceGenres.DERIVATIVE: self._EDGE_TYPE["DERIVATIVE"]}
        for rel_type in SourceGenres.get_rel_types():
            n = 0
            for tag in source_genre.genres:
                neighs = source_genre.neighbors_for_type(tag, rel_type)
                n += len(neighs)
                for neigh in neighs:
                    self._ext_graph.add_edge(source_genre.source + ":" + tag, source_genre.source + ":" + neigh,
                                             type=sg_rtype2_tr_etype[rel_type])
            print("Adding {} relations of type {}".format(n, rel_type))

    def filter_out_aliases(self, genres):
        """
            Remove aliases and replace them with their main genre from dbpedia
            (the most connected node)
        """
        assert self._ext_graph is not None
        candidates = set(genres)
        to_remove = set()
        to_add = set()
        for g in genres:
            aliases = self.get_aliases(g, norm=False)
            for alias in aliases:
                # Take the most important alias in (it would be the genre with the highest degree)
                if self._ext_graph.degree(alias) > self._ext_graph.degree(g):
                    to_add.add(alias)
                    to_remove.add(g)
                elif alias in candidates:
                    to_remove.add(alias)
        candidates -= to_remove
        candidates.update(to_add)
        return candidates

    def get_aliases(self, g, norm=False):
        """
            Return aliases for a genre
            if norm is True the aliases are normalized
        """
        assert self._ext_graph is not None
        aliases = set()
        neighbours = self.get_dbpedia_neighbours_by_relation(g, norm=norm)
        alias_relation_code = self._EDGE_TYPE["ALIAS"]
        if alias_relation_code in neighbours:
            aliases = neighbours[alias_relation_code]
        return aliases

    def get_dbpedia_neighbours_by_relation(self, g, norm=True):
        """
            Return the neighbours of a node in the graph by grouping by the relation type
        """
        result = {}
        for n, _metadata in self._ext_graph[g].items():
            # Filter out the neighbours which are normalized nodes or the source node
            # the neighbours linked to g should have only type 4
            if n == "dbpedia" or self._ext_graph.node[n]['type'] != 4:
                continue

            relation = _metadata['type']
            if relation not in result:
                result[relation] = set()

            genre = self.decode_genre(n.replace("dbpedia:", "")) if norm else n
            result[relation].add(genre)

        return result