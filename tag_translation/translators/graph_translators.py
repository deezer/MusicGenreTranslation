import numpy as np
import pandas as pd

from tag_translation.utils import read_translation_table
from tag_translation.translators.translator import LogRegTranslator


class DbpMappingTranslator(LogRegTranslator):

    def __init__(self, tag_manager, table_path):
        self.tag_manager = tag_manager
        self.source_genres = self.tag_manager.source_tags
        self.target_genres = self.tag_manager.target_tags

        self.norm_table = read_translation_table(table_path, tag_manager)
        self.W = self.norm_table.values.T
        self.b = np.zeros((1,))

    def get_translation_table(self):
        raise NotImplementedError("")

    def get_translation_table_per_source(self):
        raise NotImplementedError("")
