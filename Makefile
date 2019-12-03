#
#									  88                     ad88 88 88
#									  88                    d8"   "" 88
#									  88                    88       88
#		88,dPYba,,adPYba,  ,adPPYYba, 88   ,d8  ,adPPYba, MM88MMM 88 88  ,adPPYba,
#		88P'   "88"    "8a ""     `Y8 88 ,a8"  a8P_____88   88    88 88 a8P_____88
#		88      88      88 ,adPPPPP88 8888[    8PP"""""""   88    88 88 8PP"""""""
#		88      88      88 88,    ,88 88`"Yba, "8b,   ,aa   88    88 88 "8b,   ,aa
#		88      88      88 `"8bbdP"Y8 88   `Y8a `"Ybbd8"'   88    88 88  `"Ybbd8"'
#
#


PYTHON=python

#		d88888b d8b   db db    db
#		88'     888o  88 88    88
#		88ooooo 88V8o 88 Y8    8P
#		88~~~~~ 88 V8o88 `8b  d8'
#		88.     88  V888  `8bd8'
#		Y88888P VP   V8P    YP

DATA_DIR=data
ACOUSTICBRAINZ_DIR=$(DATA_DIR)/acousticbrainz
COMMON_RECORDINGS_DATA=$(ACOUSTICBRAINZ_DIR)/translation_dataset_all.tsv

FOLDS_DIR=$(DATA_DIR)/folds
TAXONOMY_DIR=$(DATA_DIR)/taxonomies
TAG_REP_DIR=$(DATA_DIR)/tag_representation
SCRIPTS_DIR=scripts
RESULTS_DIR=$(DATA_DIR)/results
KB_RESULTS_DIR=$(RESULTS_DIR)/kb_results
BS_RESULTS_DIR=$(RESULTS_DIR)/bs_results
ML_RESULTS_DIR=$(RESULTS_DIR)/ml_results
MAP_RESULTS_DIR=$(RESULTS_DIR)/map_results
MAP_RESULTS_DIR_NO_BIAS=$(RESULTS_DIR)/map_results_no_bias
PLOTS_DIR=$(DATA_DIR)/plots


#		d888888b  .d8b.  d8888b.  d888b  d88888b d888888b .d8888.
#		`~~88~~' d8' `8b 88  `8D 88' Y8b 88'     `~~88~~' 88'  YP
#		   88    88ooo88 88oobY' 88      88ooooo    88    `8bo.
#		   88    88~~~88 88`8b   88  ooo 88~~~~~    88      `Y8b.
#		   88    88   88 88 `88. 88. ~8~ 88.        88    db   8D
#		   YP    YP   YP 88   YD  Y888P  Y88888P    YP    `8888Y'


DBPEDIA_GRAPH=$(TAXONOMY_DIR)/dbpedia.graphml

ARTIST_SPLIT_DISCOGS=$(FOLDS_DIR)/discogs_4-fold_by_artist.tsv
ARTIST_SPLIT_LASTFM=$(FOLDS_DIR)/lastfm_4-fold_by_artist.tsv
ARTIST_SPLIT_TAGTRAUM=$(FOLDS_DIR)/tagtraum_4-fold_by_artist.tsv
ALL_ARTIST_SPLITS=$(FOLDS_DIR)/.flag
ALL_PLOTS=$(PLOTS_DIR)/.flag

AB_SOURCES_GRAPH=$(TAXONOMY_DIR)/.flag
ALL_TAG_REPS=$(TAG_REP_DIR)/.flag

TRANSLATION_TABLE_TARGET_DISCOGS=$(DATA_DIR)/translation_table_target_discogs.csv
TRANSLATION_TABLE_TARGET_LASTFM=$(DATA_DIR)/translation_table_target_lastfm.csv
TRANSLATION_TABLE_TARGET_TAGTRAUM=$(DATA_DIR)/translation_table_target_tagtraum.csv

KB_RESULTS_DISCOGS=$(KB_RESULTS_DIR)/discogs/.flag
KB_RESULTS_LASTFM=$(KB_RESULTS_DIR)/lastfm/.flag
KB_RESULTS_TAGTRAUM=$(KB_RESULTS_DIR)/tagtraum/.flag

BS_RESULTS_DISCOGS=$(BS_RESULTS_DIR)/discogs/.flag
BS_RESULTS_LASTFM=$(BS_RESULTS_DIR)/lastfm/.flag
BS_RESULTS_TAGTRAUM=$(BS_RESULTS_DIR)/tagtraum/.flag

ML_RESULTS_DISCOGS=$(ML_RESULTS_DIR)/discogs/.flag
ML_RESULTS_LASTFM=$(ML_RESULTS_DIR)/lastfm/.flag
ML_RESULTS_TAGTRAUM=$(ML_RESULTS_DIR)/tagtraum/.flag

MAP_RESULTS_DISCOGS=$(MAP_RESULTS_DIR)/discogs/.flag
MAP_RESULTS_LASTFM=$(MAP_RESULTS_DIR)/lastfm/.flag
MAP_RESULTS_TAGTRAUM=$(MAP_RESULTS_DIR)/tagtraum/.flag

MAP_RESULTS_NO_BIAS_DISCOGS=$(MAP_RESULTS_DIR_NO_BIAS)/discogs/.flag
MAP_RESULTS_NO_BIAS_LASTFM=$(MAP_RESULTS_DIR_NO_BIAS)/lastfm/.flag
MAP_RESULTS_NO_BIAS_TAGTRAUM=$(MAP_RESULTS_DIR_NO_BIAS)/tagtraum/.flag


#		.d8888.  .o88b. d8888b. d888888b d8888b. d888888b .d8888.
#		88'  YP d8P  Y8 88  `8D   `88'   88  `8D `~~88~~' 88'  YP
#		`8bo.   8P      88oobY'    88    88oodD'    88    `8bo.
#		  `Y8b. 8b      88`8b      88    88~~~      88      `Y8b.
#		db   8D Y8b  d8 88 `88.   .88.   88         88    db   8D
#		`8888Y'  `Y88P' 88   YD Y888888P 88         YP    `8888Y'

SCRIPT_DOWNLOAD_DATA=$(SCRIPTS_DIR)/download_all_data.sh
SCRIPT_ASSEMBLE_DATASET=$(SCRIPTS_DIR)/put_together_dataset.py
SCRIPT_SPLIT_DATASETS=$(SCRIPTS_DIR)/split_dataset_into_folds.py
SCRIPT_DBPEDIA_TAXONOMY=$(SCRIPTS_DIR)/dbpedia_repr_to_taxonomy.py
SCRIPT_AB_SOURCE_TAXONOMY=$(SCRIPTS_DIR)/acousticbrainz_raw_repr_to_taxonomy.py
SCRIPT_BUILD_TAG_REP=$(SCRIPTS_DIR)/build_tag_representation.py
SCRIPT_BUILD_TRANSLATION_TABLE=$(SCRIPTS_DIR)/build_translation_table.py
SCRIPT_COMPUTE_KB_RESULTS=$(SCRIPTS_DIR)/compute_kb_results.py
SCRIPT_COMPUTE_BS_RESULTS=$(SCRIPTS_DIR)/compute_bs_results.py
SCRIPT_RUN_ALL_LOGREG=$(SCRIPTS_DIR)/run_all_logreg.py
SCRIPT_PLOT_RESULTS=$(SCRIPTS_DIR)/plot_results.py

seed: $(COMMON_RECORDINGS_DATA)


taxonomies: seed $(TAXONOMY_DIR) $(DBPEDIA_GRAPH) $(AB_SOURCES_GRAPH)

tag_representations: taxonomies $(TAG_REP_DIR) $(ALL_TAG_REPS)

translation_table: tag_representations $(TRANSLATION_TABLE_TARGET_DISCOGS) $(TRANSLATION_TABLE_TARGET_LASTFM) $(TRANSLATION_TABLE_TARGET_TAGTRAUM)


$(COMMON_RECORDINGS_DATA): $(SCRIPT_DOWNLOAD_DATA) $(SCRIPT_ASSEMBLE_DATASET)
	/bin/bash $<
	python $(word 2, $^) $(ACOUSTICBRAINZ_DIR)


$(DBPEDIA_GRAPH): $(SCRIPT_DBPEDIA_TAXONOMY)
	$(PYTHON) $< $(shell dirname $@)

$(AB_SOURCES_GRAPH): $(SCRIPT_AB_SOURCE_TAXONOMY)
	$(PYTHON) $< $(ACOUSTICBRAINZ_DIR) $(shell dirname $@)
	touch $@

$(ALL_TAG_REPS): $(SCRIPT_BUILD_TAG_REP) $(AB_SOURCES_GRAPH)
	$(PYTHON) $< $(shell dirname $(word 2, $^)) $(shell dirname $@)
	touch $@


$(TRANSLATION_TABLE_TARGET_DISCOGS): $(SCRIPT_BUILD_TRANSLATION_TABLE)
	$(PYTHON) $< -s lastfm tagtraum -t discogs --tag-rep $(TAG_REP_DIR) --tax-dir $(TAXONOMY_DIR) -o $@

$(TRANSLATION_TABLE_TARGET_LASTFM): $(SCRIPT_BUILD_TRANSLATION_TABLE)
	$(PYTHON) $< -s discogs tagtraum -t lastfm --tag-rep $(TAG_REP_DIR) --tax-dir $(TAXONOMY_DIR) -o $@

$(TRANSLATION_TABLE_TARGET_TAGTRAUM): $(SCRIPT_BUILD_TRANSLATION_TABLE)
	$(PYTHON) $< -s lastfm discogs -t tagtraum --tag-rep $(TAG_REP_DIR) --tax-dir $(TAXONOMY_DIR) -o $@


$(RESULTS_DIR):
	mkdir -p $@

$(KB_RESULTS_DIR): $(RESULTS_DIR)
	mkdir -p $@

$(TAXONOMY_DIR):
	mkdir -p $@

$(TAG_REP_DIR):
	mkdir -p $@


$(KB_RESULTS_DIR)/%/.flag: $(SCRIPT_COMPUTE_KB_RESULTS) $(DATA_DIR)/translation_table_target_%.csv
	base_dir=$(shell dirname $@); \
	mkdir -p $$base_dir; \
	bname=$(shell basename $(shell dirname $@)); \
	$(PYTHON) $< -t $$bname --tr-table $(word 2, $^) -o $$base_dir
	touch $@


$(BS_RESULTS_DIR)/%/.flag: $(SCRIPT_COMPUTE_BS_RESULTS)
	base_dir=$(shell dirname $@); \
	bname=$(shell basename $(shell dirname $@)); \
	$(PYTHON) $< -t $$bname -o $(BS_RESULTS_DIR)
	touch $@


$(ML_RESULTS_DIR)/%/.flag: $(SCRIPT_RUN_ALL_LOGREG)
	base_dir=$(shell dirname $@); \
	mkdir -p $$base_dir; \
	bname=$(shell basename $(shell dirname $@)); \
	$(PYTHON) $< -t $$bname -o $$base_dir
	touch $@


$(MAP_RESULTS_DIR)/%/.flag: $(SCRIPT_RUN_ALL_LOGREG) $(DATA_DIR)/translation_table_target_%.csv
	base_dir=$(shell dirname $@); \
	mkdir -p $$base_dir; \
	bname=$(shell basename $(shell dirname $@)); \
	$(PYTHON) $< -t $$bname --tr-table $(word 2, $^) -o $$base_dir --bias-reg 0.1
	touch $@


$(MAP_RESULTS_DIR_NO_BIAS)/%/.flag: $(SCRIPT_RUN_ALL_LOGREG) $(DATA_DIR)/translation_table_target_%.csv
	base_dir=$(shell dirname $@); \
	mkdir -p $$base_dir; \
	bname=$(shell basename $(shell dirname $@)); \
	$(PYTHON) $< -t $$bname --tr-table $(word 2, $^) -o $$base_dir
	touch $@



$(ALL_ARTIST_SPLITS): $(SCRIPT_SPLIT_DATASETS) $(COMMON_RECORDINGS_DATA)
	$(PYTHON) $< $(word 2, $^) $(shell dirname $@) --folds 4 --by artist
	touch $@


$(PLOTS_DIR):
	mkdir -p $@


$(ALL_PLOTS): $(SCRIPT_PLOT_RESULTS) $(PLOTS_DIR) results
	$(PYTHON) $< $(RESULTS_DIR) $(PLOTS_DIR)
	touch $@


all_kb_results: $(ALL_ARTIST_SPLITS) $(KB_RESULTS_DISCOGS) $(KB_RESULTS_LASTFM) $(KB_RESULTS_TAGTRAUM)

all_bs_results: $(ALL_ARTIST_SPLITS) $(BS_RESULTS_DISCOGS) $(BS_RESULTS_LASTFM) $(BS_RESULTS_TAGTRAUM)

all_ml_results: $(ALL_ARTIST_SPLITS) $(ML_RESULTS_DISCOGS) $(ML_RESULTS_LASTFM) $(ML_RESULTS_TAGTRAUM)

all_map_results: $(ALL_ARTIST_SPLITS) $(MAP_RESULTS_DISCOGS) $(MAP_RESULTS_LASTFM) $(MAP_RESULTS_TAGTRAUM)

all_map_no_bias_results: $(ALL_ARTIST_SPLITS) $(MAP_RESULTS_NO_BIAS_DISCOGS) $(MAP_RESULTS_NO_BIAS_LASTFM) $(MAP_RESULTS_NO_BIAS_TAGTRAUM)

results: $(RESULTS_DIR) translation_table all_bs_results all_kb_results all_ml_results all_map_results all_map_no_bias_results

plots: $(ALL_PLOTS)

.PHONY: taxonomies tag_representations translation_table results all_kb_results all_ml_results all_map_results plots
