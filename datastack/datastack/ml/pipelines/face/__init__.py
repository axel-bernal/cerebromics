import pandas as pd
import numpy as np
import json
import math
import operator

class JsonConverter():
    @classmethod
    def split_array(self, s, title="PC"):
        l = json.loads(s)
        t = [title+str(i+1) for i in range(len(l))]
        return dict(zip(t, l))


class DatasetAssembler(object):

    def __init__(self, rdb, vdb, frr, rosetta_map=None, variantdb_map=None):

        from datastack.ml.pipelines import face
        import os

        self.rdb = rdb
        self.vdb = vdb
        self.frr = frr

        self.rosetta_map = rosetta_map
        if self.rosetta_map is None:
            self.rosetta_map = {}

        self.variantdb_map = variantdb_map
        if self.variantdb_map is None:
            self.variantdb_map = {}

        fdir = os.path.dirname(face.__file__)

        #ftp://ftp.ncbi.nlm.nih.gov/pub/CCDS/current_human/CCDS.current.txt
        self.db_ccds      = pd.read_csv(os.path.join(fdir, "data/CCDS.current.txt"), sep="\t")

        #http://www.genenames.org/cgi-bin/download?col=gd_hgnc_id&col=gd_app_sym&col=gd_app_name&col=gd_status&col=gd_prev_sym&col=gd_aliases&col=gd_name_aliases&col=gd_pub_chrom_map&col=gd_pub_acc_ids&col=gd_pub_refseq_ids&col=md_mim_id&col=md_prot_id&col=md_ensembl_id&status=Approved&status=Entry+Withdrawn&status_opt=2&where=&order_by=gd_app_sym_sort&format=text&limit=&hgnc_dbtag=on&submit=submit
        self.db_genenames = pd.read_csv(os.path.join(fdir, "data/genenames_custom.tsv"), sep="\t")


        self.db_genemap   = pd.read_csv(os.path.join(fdir, "data/mart_export.txt"), sep="\t")


    # ------------------------------------------
    #    Gene properties
    # ------------------------------------------

    def _convert_chromosome_name(self, chrin):
        if chrin in ["X", "Y"]:
            return chrin
        elif chrin == "M":
            return "MT"
        else:
            return str(int(chrin))

    def _get_gene_range(self, e_gname, num_based = 0, add_left = 0, add_right=0):

        _n      = self.db_genemap[self.db_genemap["Ensembl Gene ID"]==e_gname]
        if len(_n) == 0:
            raise Exception("Cannot find the gene")

        _start, _end, _chr  = list(_n[:1][["Gene Start (bp)", "Gene End (bp)", "Chromosome Name"]].values[0])
        _chr = self._convert_chromosome_name(_chr)
        out                 = [[_chr, _start-num_based-add_left, _end-num_based+add_right]]

        return out

    def get_gene_range(self, e_gname, num_based = 0, add_left = 0, add_right=0):
        e_gname_l = e_gname
        if type(e_gname_l)==str:
            e_gname_l =[e_gname_l]

        out_all = []
        for g in e_gname_l:
            out = self._get_gene_range(g, num_based = num_based, add_left = add_left, add_right=add_right)
            _count = 1
            for o in out:
                out_all.append(o+["{}.range.{}".format(g, _count)])
                _count+=1

        return out_all

    def _get_transcript_ranges(self, e_gname, num_based = 0, add_left = 0, add_right=0):

        _n = self.db_genemap[self.db_genemap["Ensembl Gene ID"]==e_gname]
        if len(_n) == 0:
            raise Exception("Cannot find the gene")

        _s  = _n[["Transcript Start (bp)", "Transcript End (bp)", "Chromosome Name"]]

        out = []
        for slot in _s.values:
            _chr = self._convert_chromosome_name(slot[2])
            out.append([_chr, slot[0]-num_based-add_left, slot[1]-num_based+add_right])

        return out

    def get_transcript_ranges(self, e_gname, num_based = 0, add_left = 0, add_right=0):
        e_gname_l = e_gname
        if type(e_gname_l)==str:
            e_gname_l =[e_gname_l]

        out_all = []
        for g in e_gname_l:
            out = self._get_transcript_ranges(g, num_based = num_based, add_left = add_left, add_right=add_right)
            _count = 1
            for o in out:
                out_all.append(o+["{}.transcript.{}".format(g, _count)])
                _count+=1

        return out_all

    def _get_cdna_ranges(self, e_gname, num_based = 0, add_left = 0, add_right=0):

        _n      = self.db_genenames[self.db_genenames["Ensembl ID(supplied by Ensembl)"]==e_gname]
        _n_conv = _n[u'Approved Symbol'].values[0]
        _s      = self.db_ccds[self.db_ccds[u'gene']==_n_conv][["#chromosome", "cds_locations"]]

        out = []
        for slot in _s.values:
            _chr   = self._convert_chromosome_name(slot[0])
            _parts = slot[1].replace("[", "").replace("]", "").split(",")
            for p in _parts:
                _p_start = p.split("-")[0]
                _p_end   = p.split("-")[1]
                if len(_p_start) == 0 or len(_p_end) == 0:
                    continue

                _p_start = int(_p_start)
                _p_end   = int(_p_end)
                out.append([_chr, _p_start-num_based-add_left, _p_end-num_based+add_right])

        return out

    def get_cdna_ranges(self, e_gname, num_based = 0, add_left = 0, add_right=0):
        e_gname_l = e_gname
        if type(e_gname_l)==str:
            e_gname_l =[e_gname_l]

        out_all = []
        for g in e_gname_l:
            out = self._get_cdna_ranges(g, num_based = num_based, add_left = add_left, add_right=add_right)
            _count = 1
            for o in out:
                out_all.append(o+["{}.cnda.{}".format(g, _count)])
                _count+=1

        return out_all

    # ------------------------------------------
    #    Tools
    # ------------------------------------------

    def _normalize_values(self, _xy, col_name):

        if col_name not in _xy.columns:
            return

        _t = _xy[col_name].values
        _t = _t - _t.mean(0)
        _t = _t/_t.std(0)

        _xy[col_name] = _t

    # ------------------------------------------
    #    Data extraction
    # ------------------------------------------

    def prepare_rosetta(self, key_names, **kwargs):

        # Filters
        filters           = kwargs.pop("filters", {"lab.ProjectID": "FACE", "lab.ClientSubjectID": (None, "!=")})
        normalize_keys    = kwargs.pop("normalize_keys", [])
        self.max_samples  = kwargs.pop("max_samples", None)
        force_test_set    = kwargs.pop("force_test_set", None)

        # X Keys
        self.rdb_keys      = []

        for k in key_names:
            if k in self.rosetta_map.keys():
                self.rdb_keys = self.rdb_keys+self.rosetta_map[k]
            else:
                raise Exception("Cannot find the key map: {}",format(k))
        
        self.rdb_key_names        = key_names
        self.rdb_filters          = filters
        self.rdb_normalize_keys   = normalize_keys

        index_keys  = ["lab.ClientSubjectID", "ds.index.sample_key"]

        # Y values
        femb_keys   = self.rdb.find_keys("dynamic.FACE.face.{}_visit1.DepthPC".format(self.frr.db_version))
        femc_keys   = self.rdb.find_keys("dynamic.FACE.face.{}_visit1.ColorPC".format(self.frr.db_version))

        # Find samples set for the case of enforicing the
        if force_test_set:

            temp_filters                    = filters
            temp_filters["lab.ClientSubjectID"] = [(force_test_set, "!in"),(None, "<>")]

            new_samples   = self.rdb.query(filters   = temp_filters,
                                           keys        = ["lab.ClientSubjectID"],
                                           with_nans   = False
                                         )

            new_samples                = list(new_samples["lab.ClientSubjectID"].values)
            filters["lab.ClientSubjectID"] = [(force_test_set+new_samples, "in")]

        self.r_xy   = self.rdb.query(filters    = filters,
                                     keys       = self.rdb_keys + index_keys + femb_keys  +femc_keys,
                                     with_nans  = False
                                     )

        _rep = self.r_xy[self.r_xy["lab.ClientSubjectID"].duplicated()]
        if len(_rep)>0:
            raise Exception("One or more samples are repeated: {}".format(_rep["lab.ClientSubjectID"].values))

        self.r_xy.index = self.r_xy["lab.ClientSubjectID"]
        del self.r_xy["lab.ClientSubjectID"]

        # Normalize
        for k in normalize_keys:
            self._normalize_values(self.r_xy, k)

        # Limit samples
        if self.max_samples is not None:
            if force_test_set is None:
                self.r_xy = self.r_xy[:self.max_samples]
            else:
                _outside = list(self.r_xy[~self.r_xy.index.isin(force_test_set)].index)[:self.max_samples]
                self.r_xy = self.r_xy[self.r_xy.index.isin(force_test_set+_outside)]

        # Store samples
        self.tot_samples      = len(self.r_xy)
        self.selected_samples = self.r_xy["ds.index.sample_key"].values
        del self.r_xy["ds.index.sample_key"]
        
        self.xy               = self.r_xy
        self.model_keys       = self.rdb_keys

        return self.tot_samples, len(self.r_xy.columns)

    # ------------------------------------------
    #    Variant extraction
    # ------------------------------------------

    def _build_dict_results(self, ranges, samples):

        r_variants = {}

        for N in range(len(ranges)):
            filters              = {}
            filters['samples']   = samples
            filters['variants']  = "{}_{}-{}".format(ranges[N][0], ranges[N][1], ranges[N][2])

            keys                 = ["variantcall"]

            _tt = self.vdb.query(filters, keys)
            if len(_tt.columns)>0:
                r_variants[ranges[N][3]] = _tt

        return r_variants


    def _imputation_and_maf(self, r_variants, maf=None, std=False):

        r_variants_clean = {}

        for k in r_variants.keys():

            _tdf = r_variants[k]

            _tdf = _tdf[_tdf >= 0]
            _tdf = _tdf.fillna(_tdf.mean())
            _tdf = _tdf.fillna(0.0)

            _len = len(_tdf.columns)

            if maf is not None:
                _tdf = _tdf[_tdf.columns[_tdf.sum()/len(_tdf) > maf]]

            if std:
                _tdf = (_tdf - _tdf.mean())/(_tdf.max()-_tdf.min()+0.00001)

            if len(_tdf.columns)>0:
                r_variants_clean[k] = _tdf

        return r_variants_clean

    def _run_pca(self, r_variants, pca_components=10):

        from sklearn.decomposition import PCA

        r_variants_pc = {}

        for k in r_variants.keys():

            _tdf = r_variants[k]
            _n_components = min(pca_components, len(_tdf.columns))
            if _n_components>=len(_tdf.columns):
              pca  = PCA(n_components=_n_components)
              pca.fit(_tdf)
              _tdf = pca.transform(_tdf)
            
            r_variants_pc[k] = _tdf

        return r_variants_pc

    def add_variants(self, key_names, operation, run_pca = False, pca_components=10, maf=None, num_based = 0, add_left = 0, add_right=0):

        import numpy as np

        if operation not in ["get_transcript_ranges", "get_gene_range", "get_cdna_ranges"]:
            raise Exception("Operation not supported")

        self.ranges = []
        for k in key_names:
            if k in self.variantdb_map.keys():
                _ranges =  getattr(self, operation)(self.variantdb_map[k], num_based = num_based, add_left = add_left, add_right=add_right)
                self.ranges = self.ranges + _ranges
            else:
                raise Exception("Cannot find the key map: {}".format(k))

        self.vdb_key_names = key_names
        self.vdb_operation = operation
        self.vdb_params    = {"run_pca":run_pca, "pca_components":pca_components, "maf":maf, "num_based":num_based, "add_left":add_left, "add_right":add_right}

        r_variants       = self._build_dict_results(self.ranges, self.selected_samples)

        if run_pca:
            r_variants_clean = self._imputation_and_maf(r_variants, maf=maf, std=True)
            r_variants_final = self._run_pca(r_variants_clean, pca_components=pca_components)

        else:
            r_variants_final = self._imputation_and_maf(r_variants, maf=maf)

        # Create final v_xy
        _all      = r_variants_final.keys()
        self.v_xy = r_variants_final[_all[0]]

        for N in range(len(_all)-1):
            self.v_xy = np.hstack((self.v_xy, r_variants_final[_all[N+1]]))

        self.v_xy         = pd.DataFrame(self.v_xy, index=self.r_xy.index)
        self.v_xy.columns = ["vdb.{}".format(c) for c in self.v_xy.columns]

        self.vdb_keys     = list(self.v_xy.columns)

        # Merge
        self.xy         = self.r_xy.merge(self.v_xy, left_index=True, right_index=True)
        self.model_keys = self.rdb_keys + self.vdb_keys

        return len(self.xy), len(self.v_xy.columns)



class TrainTestContainer(object):

    def __init__(self, da):
        self.da = da

    def _face_PC(self, N, embt="DepthPC"):
        return ["dynamic.FACE.face.{}_visit1.{}{}".format(self.da.frr.db_version, embt, i+1) for i in range(N)]

    def _genome_PC(self, N):
        return ["dynamic.FACE.genome.v1.pc{}".format(i+1) for i in range(N)]

    def create_train_test_split(self, heldout_samples = None, remove_samples = None,
                                train_test_ratio = 0.93, N_face_PC = 40, force_test_set=None):

        self.heldout_samples  = heldout_samples
        self.remove_samples   = remove_samples
        self.train_test_ratio = train_test_ratio
        self.N_face_PC        = N_face_PC
        self.force_test_set   = force_test_set


        _depth_keys    = "dynamic.FACE.face.{}_visit1.DepthPC".format(self.da.frr.db_version)
        _color_keys    = "dynamic.FACE.face.{}_visit1.ColorPC".format(self.da.frr.db_version)

        idx_to_select  = np.array([True]*len(self.da.xy.index))

        if self.heldout_samples is not None:
            idx_in_heldout = self.da.xy.index.isin(self.heldout_samples)
            idx_to_select  = idx_to_select & ~idx_in_heldout


        if self.remove_samples is not None:
            idx_to_select  = idx_to_select & ~self.da.xy.index.isin(self.remove_samples)

        self.X         = self.da.xy[idx_to_select][self.da.model_keys]
        self.X_samples = self.da.selected_samples[idx_to_select]
        self.Y_depth   = self.da.xy[idx_to_select][_depth_keys].apply(lambda s: pd.Series(JsonConverter.split_array(s, title=_depth_keys)))
        self.Y_color   = self.da.xy[idx_to_select][_color_keys].apply(lambda s: pd.Series(JsonConverter.split_array(s, title=_color_keys)))

        if heldout_samples is not None:
            self.X_heldout             = self.da.xy[idx_in_heldout][self.da.model_keys]
            self.Y_depth_heldout       = self.da.xy[idx_in_heldout][_depth_keys].apply(lambda s: pd.Series(JsonConverter.split_array(s, title=_depth_keys)))
            self.Y_color_heldout       = self.da.xy[idx_in_heldout][_color_keys].apply(lambda s: pd.Series(JsonConverter.split_array(s, title=_color_keys)))
            self.test_y_depth_heldout  = self.Y_depth_heldout[self._face_PC(N_face_PC, embt="DepthPC")]
            self.test_y_color_heldout  = self.Y_color_heldout[self._face_PC(N_face_PC, embt="ColorPC")]

        if self.force_test_set:

            self.train_x         = self.X[~self.X.index.isin(force_test_set)]
            self.train_x_samples = self.X_samples[~self.X.index.isin(force_test_set)]
            self.train_y_depth   = self.Y_depth[~self.X.index.isin(force_test_set)][self._face_PC(N_face_PC, embt="DepthPC")]
            self.train_y_color   = self.Y_color[~self.X.index.isin(force_test_set)][self._face_PC(N_face_PC, embt="ColorPC")]

            self.test_x          = self.X[self.X.index.isin(force_test_set)]
            self.test_x_samples  = self.X_samples[self.X.index.isin(force_test_set)]
            self.test_y_depth    = self.Y_depth[self.X.index.isin(force_test_set)][self._face_PC(N_face_PC, embt="DepthPC")]
            self.test_y_color    = self.Y_color[self.X.index.isin(force_test_set)][self._face_PC(N_face_PC, embt="ColorPC")]
        else:

            self.train_x         = self.X[:int(len(self.X)*self.train_test_ratio)]
            self.train_x_samples = self.X_samples[:int(len(self.X)*self.train_test_ratio)]
            self.train_y_depth = self.Y_depth[:int(len(self.X)*self.train_test_ratio)][self._face_PC(N_face_PC, embt="DepthPC")]
            self.train_y_color = self.Y_color[:int(len(self.X)*self.train_test_ratio)][self._face_PC(N_face_PC, embt="ColorPC")]

            self.test_x          = self.X[int(len(self.X)*self.train_test_ratio):]
            self.test_x_samples  = self.X_samples[int(len(self.X)*self.train_test_ratio):]
            self.test_y_depth    = self.Y_depth[int(len(self.X)*self.train_test_ratio):][self._face_PC(N_face_PC, embt="DepthPC")]
            self.test_y_color    = self.Y_color[int(len(self.X)*self.train_test_ratio):][self._face_PC(N_face_PC, embt="ColorPC")]


        assert len(self.X)         == len(self.train_x) + len(self.test_x)
        assert len(self.X_samples) == len(self.train_x_samples) + len(self.test_x_samples)
        assert len(self.Y_depth)   == len(self.train_y_depth) + len(self.test_y_depth)
        assert len(self.Y_color)   == len(self.train_y_color) + len(self.test_y_color)

        return len(self.train_x), len(self.test_x)

