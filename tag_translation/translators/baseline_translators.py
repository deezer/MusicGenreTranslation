import numpy as np
from nltk.metrics import edit_distance
from tag_translation.translators.translator import LogRegTranslator
from tag_translation.utils import load_genre_representation

class BaselineTranslator(LogRegTranslator):
    def __init__(self, tag_manager):
        self.tag_manager = tag_manager
        self.W = self.compute_distances()
        self.b = np.zeros((1,))

    def get_distance(self, str1, str2):
        raise NotImplementedError("")

    def get_source_tags(self):
        raise NotImplementedError("")

    def get_target_tags(self):
        raise NotImplementedError("")

    def compute_distances(self):
        s_genres = self.get_source_tags()
        t_genres = self.get_target_tags()
        ns = len(s_genres)
        nt = len(t_genres)
        d = np.zeros((ns, nt))
        for i in range(ns):
            for j in range(nt):
                d[i, j] = self.get_distance(s_genres[i], t_genres[j])
        return d.T


class LevenshteinTranslator(BaselineTranslator):

    def get_distance(self, str1, str2):
        return 1 - edit_distance(str1, str2) / max(len(str1), len(str2))

    def get_source_tags(self):
        return self.tag_manager.unprefixed_source_tags

    def get_target_tags(self):
        return self.tag_manager.unprefixed_target_tags

class NormDirectTranslator(BaselineTranslator):

    def __init__(self, tag_manager, input_dir):
        self.tag_rep, _, _, _ = load_genre_representation(input_dir)
        super().__init__(tag_manager)

    def get_distance(self, str1, str2):
        return int(str1 == str2)

    def get_source_tags(self):
        return [self.tag_rep.decode_genre(t) for t in self.tag_manager.unprefixed_source_tags]

    def get_target_tags(self):
        return [self.tag_rep.decode_genre(t) for t in self.tag_manager.unprefixed_target_tags]
