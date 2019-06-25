import os
opj = os.path.join
opd = os.path.dirname
opa = os.path.abspath

BASE_DATA_DIR=opj(opd(opd(opa(__file__))), "data")
EXTENDED_GRAPH_PATH=opj(BASE_DATA_DIR, "tag_representation/extended_graph.graphml")
FOLDS_DIR=opj(BASE_DATA_DIR, "folds")
N_JOBS=4
SOLVER="lbfgs"
TEST_DATA_DIR = opj(BASE_DATA_DIR, "test_data")
RANDOM_STATE=0
FRAC_MAX=7  # We will train only on up to a fraction of 2**-(FRAC_MAX+1) of the data per fold.