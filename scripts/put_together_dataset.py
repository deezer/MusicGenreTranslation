import os
import sys
import pandas as pd


ope = os.path.exists
opj = os.path.join


def read_and_format(path, source):
    print(f"processing {path} {source}")
    t = pd.read_csv(path, sep="\t", index_col="recordingmbid")
    all_tags = []
    genre_columns = [c for c in t.columns if "genre" in c]
    for ts in zip(*[t[key] for key in genre_columns]):
        all_tags.append([g.split("---")[-1] for g in ts if isinstance(g, str)])
    t[source] = all_tags
    for key in genre_columns:
        del t[key]
    return t


if __name__ == "__main__":
    acoustic_brainz_dir = sys.argv[1]
    assert ope(opj(acoustic_brainz_dir, "recordingmbid2artistmbid.tsv"))
    sources = ["discogs", "lastfm", "tagtraum"]
    files = []
    tables = []
    for t in sources:
        files.append((opj(acoustic_brainz_dir, f"ab_mediaeval_train_{t}.tsv"), t))
        files.append((opj(acoustic_brainz_dir, f"ab_mediaeval_val_{t}.tsv"), t))

    for f in files:
        assert ope(f[0])

    artist_mapping = pd.read_csv(opj(acoustic_brainz_dir, "recordingmbid2artistmbid.tsv"), sep="\t",
                                 index_col="recordingmbid")
    table_per_source = {}
    for f in files:
        table = read_and_format(*f)
        if f[1] not in table_per_source:
            table_per_source[f[1]] = table
        else:
            table_per_source[f[1]] = pd.concat([table_per_source[f[1]], table])
    all_tables = list(table_per_source.values())
    final = all_tables[0]
    for t in all_tables[1:]:
        print(t.columns)
        final = final.join(t, lsuffix="_left", rsuffix="_right", how="outer")
        print(final.columns)
        final["releasegroupmbid"] = final["releasegroupmbid_left"]
        final["releasegroupmbid"][final["releasegroupmbid"].isnull()] = final["releasegroupmbid_right"]
        del final["releasegroupmbid_left"]
        del final["releasegroupmbid_right"]

    nb = len(final)
    final = final.join(artist_mapping, how="inner")
    rem = len(final)
    if rem < nb:
        print("WARNING: some recordingmbids haven't been matched to an artistmbid")

    final = final[["releasegroupmbid", "artistmbid", "discogs", "lastfm", "tagtraum"]]
    final.to_csv(opj(acoustic_brainz_dir, "translation_dataset_all.tsv"), sep="\t")
