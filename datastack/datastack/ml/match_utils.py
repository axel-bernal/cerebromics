from datastack.dbs.rdb import RosettaDBMongo
try:
    from fuzzywuzzy import process
except:
    print "Warning: fuzzywuzzy not installed"
import pandas as pd
import numpy as np
import re
import rosetta_settings as ml_vars


def get_face_PCs(rdb, version="v1"):
    depth_col = 'dynamic.FACE.face.%s_visit1.DepthPC' % version
    color_col = 'dynamic.FACE.face.%s_visit1.ColorPC' % version
    Y = rdb.query(keys=[depth_col, color_col, ml_vars.SAMPLE_KEY]).dropna()
    Y.set_index(ml_vars.SAMPLE_KEY, inplace=True)

    # Convert JSON column to a dataframe which we will join
    facepcs_obs = pd.DataFrame(map(eval, Y.loc[:, depth_col]), index=Y.index)
    colorpcs_obs = pd.DataFrame(map(eval, Y.loc[:, color_col]), index=Y.index)

    facepcs_obs.columns = ['%s.%s' % (depth_col, xx + 1)
                           for xx in range(facepcs_obs.shape[1])
                           ]

    colorpcs_obs.columns = ['%s.%s' % (color_col, xx + 1)
                            for xx in range(colorpcs_obs.shape[1])
                            ]
    return pd.concat([facepcs_obs, colorpcs_obs], axis=1)


def get_v1_v2_cols(namespace=ml_vars.NAMESPACE):
    db = RosettaDBMongo(host=ml_vars.ROSETTA_URL)
    db.initialize(namespace=namespace)
    keys = db.find_keys("dynamic.FACE")
    all_visit1 = [xx for xx in keys if "visit1" in xx and "FACE_P" not in xx]
    all_visit2 = [xx for xx in keys if "visit2" in xx and xx.replace("visit2", "visit1") in all_visit1 and "sessionId" not in xx]
    df = pd.DataFrame({"visit1": [xx.replace('visit2', 'visit1') for xx in all_visit2], "visit2": all_visit2})
    df["Group"] = df["visit1"].str.extract("FACE\.(.*v[0-9]+)")
    df.set_index("visit1", inplace=True)
    df.columns = ["Best Match", "Group"]
    df.loc[df["Best Match"].str.contains("mirror"), "Group"] = df.ix[df["Best Match"].str.contains("mirror"), "Group"] + "_mirror"
    
    df["Score"] = 100
    return df


def match_obs_pred_cols(obs_cols, pred_cols):
    # Take advantage of straightforward matches, otherwise do fuzzy matching.
    matches = pd.DataFrame(index=pred_cols, columns=["Best Match", "Score"])
    matches["Score"] = 0
    for pp in pred_cols:
        # If it conforms to the spec, don't bother with fuzzy matching
        if pp.replace("FACE_P", "FACE") in obs_cols:
            matches.loc[pp, :] = [pp.replace("FACE_P", "FACE"), 100]
        else:
            # Otherwise match within variable group (face/lmdist/eye, etc.)
            group = pp.split(".")[2]
            searchset = [xx for xx in obs_cols if group in xx]
            res = process.extract(pp, searchset, limit=1)
            if len(res) > 0:
                matches.loc[pp, :] = res[0]
    print matches[matches["Score"] < 100]

    matches.index = pred_cols
    matches.columns = ["Best Match", "Score"]

    # Manually handle a couple known mapping issues
    matches.loc['dynamic.FACE_P.eyecolor.v1.l', ["Best Match", "Score"]] = ['dynamic.FACE.eyecolor.v1_visit1.l', 100]
    matches.loc['dynamic.FACE_P.eyecolor.v1.a', ["Best Match", "Score"]] = ['dynamic.FACE.eyecolor.v1_visit1.a', 100]
    matches.loc['dynamic.FACE_P.eyecolor.v1.b', ["Best Match", "Score"]] = ['dynamic.FACE.eyecolor.v1_visit1.b', 100]
    matches.loc['dynamic.FACE_P.height.v1.value', ["Best Match", "Score"]] = ['pheno.height', 100]
    matches = matches[matches["Best Match"].notnull()]
    assert matches["Best Match"].isin(obs_cols).all(), matches[~matches["Best Match"].isin(obs_cols)]

    # Print out all the cases where you had to use fuzzy matching (now usually zero)
    print "Filtering the values below:"
    print matches[matches["Score"] < 95]
    best_matches = matches[matches["Score"] >= 95]
    best_matches.loc[:, "Group"] = map(lambda xx: re.search("FACE_P\.(.*?\.(?:v[0-9+])?)", xx).groups(1), best_matches.index)
    return best_matches


def get_select_data(use_local=False, namespace=ml_vars.NAMESPACE):
    if use_local:
        return (
            pd.read_table("data_local/select_df.txt", sep="\t", index_col=0),
            pd.read_table("data_local/pred_to_obs_colnames.txt", sep="\t", index_col=0)
        )
    else:
        rdb = RosettaDBMongo(host=ml_vars.ROSETTA_URL)
        rdb.initialize(namespace=namespace)

        keys = [kk for kk in rdb.find_keys("(.*FACE)|(pheno.HLI_CALC)", regex=True) 
                if "visit2" not in kk]

        # Partition FACE column into observed and predicted
        obs_cols = [xx for xx in keys if "FACE." in xx] + ['pheno.height']
        pred_cols = [xx for xx in keys if "FACE_P" in xx]
        best_matches = match_obs_pred_cols(obs_cols, pred_cols)
        # Print the number of variables in each column set
        print best_matches["Group"].value_counts()
        # TODO pick out most recent versions of each variable
        body_group = [
                            "MalePatternBaldness.v3 ", "bmi.v1", "age.v1",
                            "gender.v1", "height.v1", "weight.v1"
                    ]
        print "Setting %s to 'body'" % ",".join(body_group)
        best_matches.ix[best_matches["Group"].isin(body_group), "Group"] = "body"
        # Retrieve data from Rosetta, applying appropriate columns
        df = rdb.query(keys=list(best_matches.index) + 
                            list(best_matches["Best Match"]) + 
                            [ml_vars.SAMPLE_KEY, ml_vars.PROJECT_KEY],
                       filters={ml_vars.PROJECT_KEY: "FACE"
                                })

        df.reset_index(inplace=True)

        # Fix predicted gender values so they match format for observed
        df["dynamic.FACE_P.gender.v1.value"] = df["dynamic.FACE_P.gender.v1.value"].replace({"Female": 0,
                                                                                             "Male": 1,
                                                                                             "Bad": np.nan,
                                                                                             "XXY": np.nan}).astype(float)

        best_matches.to_csv("data_local/pred_to_obs_colnames.txt", sep="\t")

        df.to_csv("data_local/df_pred_obs.txt", sep="\t")
        return df, best_matches


if __name__ == "__main__":
    df, best_matches = get_select_data(use_local=False, N_pcs=20)
