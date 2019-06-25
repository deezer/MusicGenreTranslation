import networkx as nx


class SourceGenres(object):
    """
        Class to represent genres per source
        Different type of information could be expected such as subgenres, aliases etc.
    """

    # More relations are supposed to be added here apart from genre-subgenre
    SUBGENRES = "subgenres"
    STORIGINS = "stylisticOrigin"
    DERIVATIVE = "derivative"
    ALIASES = "aliases"
    IS_ROOT = "is_root"

    def __init__(self, source):
        self._source = source
        self._di_graph = nx.DiGraph(source=source)

    @classmethod
    def get_rel_types(cls):
        return [cls.SUBGENRES, cls.STORIGINS, cls.DERIVATIVE, cls.ALIASES]

    @property
    def source(self):
        return self._source

    @property
    def genres(self):
        return list(self._di_graph.nodes)

    @classmethod
    def from_file(cls, path):
        graph = nx.read_graphml(path)
        sg = SourceGenres(graph.graph["source"])
        sg._di_graph = graph
        return sg

    def add_genre(self, genre):
        self._di_graph.add_node(genre)

    def add_relation(self, genre1, genre2, relation_type):
        self._di_graph.add_edge(genre1, genre2, rel_type=relation_type)

    def add_subgenre(self, genre, subgenre):
        self.add_relation(genre, subgenre, self.SUBGENRES)

    def add_stylistic_origin(self, genre, stylistic_origin_of):
        self.add_relation(genre, stylistic_origin_of, self.STORIGINS)

    def add_derivative(self, genre, derivative_from):
        self.add_relation(derivative_from, genre, self.DERIVATIVE)

    def add_alias(self, genre, alias):
        self.add_relation(genre, alias, self.ALIASES)

    def neighbors_for_type(self, genre, rel_type):
        out = []
        for node in self._di_graph.successors(genre):
            edge_data = self._di_graph.get_edge_data(genre, node)
            if edge_data["rel_type"] == rel_type:
                out.append(node)
        return out

    def get_subgenres(self, genre):
        return self.neighbors_for_type(genre, self.SUBGENRES)

    def get_origins(self, genre):
        return self.neighbors_for_type(genre, self.STORIGINS)

    def get_derivatives(self, genre):
        return self.neighbors_for_type(genre, self.DERIVATIVE)

    def get_aliases(self, genre):
        return self.neighbors_for_type(genre, self.ALIASES)

    def is_root(self, genre):
        return self._di_graph.in_degree(genre) == 0

    def get_di_graph_repr(self):
        return self._di_graph

    def save_graphml(self, outname):
        nx.write_graphml(self._di_graph, outname)