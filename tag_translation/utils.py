import os
import logging
from time import time
import pickle

import pandas as pd
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer

from tag_translation.conf import EXTENDED_GRAPH_PATH, FOLDS_DIR, TEST_DATA_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

GRAPH = None


def load_tag_csv(path, sources=("lastfm", "discogs", "tagtraum"), sep='\t'):
    # df = pd.read_csv(path, engine='python', sep=sep, nrows=nrows)
    df = pd.read_csv(path, sep=sep)

    def load_row(r):
        if isinstance(r, float):
            return []
        else:
            return eval(r)
    for source in sources:
        df[source] = df[source].apply(load_row)
    return df


def format_row_gather_tags(row):
    return [f"{source}:{tag}" for source in row.keys() for tag in row[source]]


def format_tags_for_source(tags, source):
    return [f"{source}:{t}" for t in tags]


def format_dataset_rows_and_split(df, sources, target):
    start = time()
    # train_data = df[sources].apply(format_row_gather_tags, axis=1)
    # target_data = df[[target]].apply(format_row_gather_tags, axis=1)

    train_data = []
    target_data = []
    for t in zip(*[df[s] for s in sources + [target]]):
        if len(sources) == 2:
            stags = t[:2]
        else:
            stags = t[:1]
        ttags = t[-1]
        train_data.append([])
        for i, s in enumerate(sources):
            train_data[-1].extend(format_tags_for_source(stags[i], s))
        target_data.append(format_tags_for_source(ttags, target))
    end = time()
    print("Prepared split in {:.2f} seconds".format(end-start))
    return train_data, target_data


def read_train_data(path, sources, target):
    df = pd.read_csv(path, sep=",")
    for s in sources + [target]:
        df[s] = df[s].apply(eval)
    return format_dataset_rows_and_split(df, sources, target), df


def read_translation_table(path, tag_manager):
    kb_tr_table = pd.read_csv(path, index_col=0)
    kb_tr_table = kb_tr_table[tag_manager.mlb_target.classes_]
    kb_tr_table = kb_tr_table.reindex(tag_manager.mlb_all_sources.classes_)
    return kb_tr_table


def get_tags_for_source(source):
    global GRAPH
    if GRAPH is None:
        GRAPH = nx.read_graphml(EXTENDED_GRAPH_PATH)
    nodes = set(GRAPH.neighbors(source))
    return [n.split(":")[1] for n in nodes]


def get_default_fold_split_file_for_target(target, test=False):
    if test:
        return os.path.join(TEST_DATA_DIR, "{0}_4-fold_by_artist.tsv".format(target))
    return os.path.join(FOLDS_DIR, "{0}_4-fold_by_artist.tsv".format(target))


class TagManager:
    _MANAGERS_ = {}

    def __init__(self, sources, target):
        self._sources = sources
        self._target = target
        self.source_tags = [f"{source}:{el}" for source in sources for el in get_tags_for_source(source)]
        self.target_tags = [f"{target}:{el}" for el in get_tags_for_source(target)]
        self._mlb_all_sources = None
        self._mlb_target = None

    @classmethod
    def get(cls, sources, target):
        sources_key = " ".join(sorted(sources))
        if sources_key not in cls._MANAGERS_ or target not in cls._MANAGERS_[sources_key]:
            m = TagManager(sources, target)
            cls._MANAGERS_.setdefault(sources_key, {})
            cls._MANAGERS_[sources_key][target] = m
        return cls._MANAGERS_[sources_key][target]

    @property
    def unprefixed_source_tags(self):
        return [t.split(":")[1] for t in self.source_tags]

    @property
    def unprefixed_target_tags(self):
        return [t.split(":")[1] for t in self.target_tags]

    @property
    def sources(self):
        return self._sources

    @property
    def target(self):
        return self._target

    @property
    def mlb_all_sources(self):
        if self._mlb_all_sources is None:
            self._mlb_all_sources = MultiLabelBinarizer(classes=self.source_tags, sparse_output=True)
            self._mlb_all_sources.fit([[]])
        return self._mlb_all_sources

    @property
    def mlb_target(self):
        if self._mlb_target is None:
            self._mlb_target = MultiLabelBinarizer(classes=self.target_tags, sparse_output=True)
            self._mlb_target.fit([[]])
        return self._mlb_target

    @staticmethod
    def format_tags_for_source(tags, source):
        return [f"{source}:{t}" for t in tags]

    def transform_for_target(self, df, as_array=False):
        if as_array:
            return self.mlb_target.transform(df).toarray().astype("float32")
        else:
            return self.mlb_target.transform(df)

    def transform_for_sources(self, df, as_array=False):
        if as_array:
            return self.mlb_all_sources.transform(df).toarray().astype("float32")
        else:
            return self.mlb_all_sources.transform(df)


class DataHelper:

    def __init__(self, tag_manager, dataset_path=None, train_data_path=None, eval_data_path=None):
        if dataset_path is not None:
            if train_data_path is not None:
                logger.warning("train_data_path shadows dataset_path")
            if eval_data_path is not None:
                logger.warning("eval_data_path shadows dataset_path")
            if train_data_path is not None and eval_data_path is not None:
                dataset_path = None
        self.tag_manager = tag_manager
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.dataset_path = dataset_path
        self.dataset_df = None
        if self.dataset_path is not None:
            self.load_dataset()

    def load_dataset(self):
        print("Loading dataset...")
        start = time()
        assert self.dataset_path is not None
        dst = load_tag_csv(self.dataset_path)
        print("Loaded.")
        prev_len = len(dst)
        print("Filtering...")
        cols = self.tag_manager.sources + [self.tag_manager.target] + ["fold"]
        rows = []
        for t in zip(*[dst[s] for s in cols]):
            if len(self.tag_manager.sources) == 2:
                t1, t2, t3, f = t
            else:
                t1, t3, f= t
                t2 = 0
            if len(t1) + len(t2) == 0 or len(t3) == 0:
                continue
            rows.append((t1, t2, t3, f))
        self.dataset_df = pd.DataFrame(rows, columns=cols)
        end = time()
        print(f"Kept {len(self.dataset_df)} on {prev_len} initial rows in")
        print("Took {:.2f} seconds".format(end-start))

    def transform_sources_and_target_data(self, source_df, target_df, as_array=True):
        out = self.tag_manager.transform_for_sources(source_df, as_array=as_array), \
              self.tag_manager.transform_for_target(target_df, as_array=as_array)
        return out

    def _get_dataset_split(self, data_path=None, fold=None, test=False, as_array=True, frac=1, store_train_df=False):
        if data_path is None and fold is None:
            raise ValueError("When giving a full dataset to DataHelper, you must provide the fold index "
                             "to get the training or test data.")
        if self.dataset_path is None and frac != 1:
            logger.warning("frac argument is ignored unless the dataset_path argument is provided")
        sources = self.tag_manager.sources
        target = self.tag_manager.target
        if data_path is not None:
            (source_data, target_data), df = read_train_data(data_path, sources, target)
            if store_train_df:
                self.train_df = df
            return self.transform_sources_and_target_data(source_data, target_data, as_array=as_array)
        else:
            if test:
                bool_index = self.dataset_df.fold == fold
            else:
                bool_index = self.dataset_df.fold != fold

            if frac != 1:
                df = self.dataset_df[bool_index].sample(frac=frac, random_state=RANDOM_STATE)
            else:
                df = self.dataset_df[bool_index]
            if store_train_df:
                self.train_df = df
            train_data, target_data = format_dataset_rows_and_split(df, sources, target)
            return self.transform_sources_and_target_data(train_data, target_data, as_array=as_array)

    def get_train_data(self, fold=None, as_array=True, frac=1, store_train_df=False):
        return self._get_dataset_split(self.train_data_path, fold=fold, as_array=as_array, frac=frac,
                                       store_train_df=store_train_df)

    def get_test_data(self, fold=None, as_array=False, frac=1):
        return self._get_dataset_split(self.eval_data_path, fold=fold, test=True, as_array=as_array, frac=frac)


def load_genre_representation(input_dir):
    """
        Deserialize the tag representation object and the graphs
    """
    genre_rep_file = os.path.join(input_dir, "genre_rep")
    normalized_graph_file = os.path.join(input_dir, "graph.graphml")
    extended_graph_file = os.path.join(input_dir, "extended_graph.graphml")
    dbpedia_graph_file = os.path.join(input_dir, "dbpedia.graphml")

    with open(genre_rep_file, 'rb') as pickle_file:
        tag_rep = pickle.load(pickle_file)

    norm_graph = nx.read_graphml(normalized_graph_file)
    ext_graph = nx.read_graphml(extended_graph_file)
    dbpedia_graph = nx.read_graphml(dbpedia_graph_file)

    return tag_rep, norm_graph, ext_graph, dbpedia_graph


def dump_object(obj, out_path):
    with open(out_path, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)
