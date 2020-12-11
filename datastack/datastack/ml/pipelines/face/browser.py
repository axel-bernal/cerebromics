from flask import Flask
from datastack.common.exceptions import SdkError
from flask import render_template
import os
import json
from datastack.common.settings import get_logger
from datastack.ml.pipelines.face import settings

logger = get_logger('default')

app  = Flask(__name__)
app.debug = True
_bdir = os.getcwd()

# -------------------------------
#   Scan models
# -------------------------------
def _get_model_dir(p, name):
    return os.path.join(_bdir, p, name, settings.MODEL_ROOT_PREFIX)

def _get_info_model_run(project, model, r):

    _model_dir = _get_model_dir(project, model)
    if not os.path.isfile(os.path.join(_model_dir, r, "info.json")):
        return None
    try:
        with open(os.path.join(_model_dir, r, "info.json")) as f:
            return json.load(f)
    except:
        logger.info("Cannot load the run {r}".format())
        return None

def build_model_dict():

    all_models = {}
      
    _pdir = [o for o in os.listdir(_bdir) if os.path.isdir(os.path.join(_bdir,o))]

    for p in _pdir:

      _dir     = os.path.join(_bdir, p)
      _project = p
      _subdirs = [o for o in os.listdir(_dir) if os.path.isdir(os.path.join(_dir,o))]

      # Search inside all dirs
      for m in _subdirs:
        
        if os.path.isfile(os.path.join(_dir, m, "model_info.py")):
            all_models[m] = {}

            # Search if a face pipeline run has been executed
            _model_dir = _get_model_dir(p, m)
            if not os.path.isdir(os.path.join(_model_dir)):
                continue

            # Search all the runs done
            _runs      = [o for o in os.listdir(_model_dir) if os.path.isdir(os.path.join(_model_dir,o)) ]
            
            # Scan all the runs
            for r in _runs:
                minfo = _get_info_model_run(p, m, r)
                if p=="nn": print minfo
                minfo['project'] = _project
                all_models[m][r] = minfo



    return all_models

@app.route('/')
def hello(name=None):

    import numpy

    model_list = build_model_dict()

    bar_data_color  = []
    bar_data_depth  = []
    bar_ticks       = []

    i = 0
    for d in model_list.keys():
        for r in model_list[d].keys():
            bar_data_depth.append(model_list[d][r]['results']['depth']['precision'])
            bar_data_color.append(model_list[d][r]['results']['color']['precision'])
            i  = i+1

    bar_data_depth, bar_data_color = zip(*sorted(zip(bar_data_depth, bar_data_color)))
    bar_data_depth_l = []
    bar_data_color_l = []
    bar_ticks        = []
    for i in range(len(bar_data_depth)):
      bar_data_depth_l.append([i, bar_data_depth[i]])
      bar_data_color_l.append([i, bar_data_color[i]]) 
      bar_ticks.append([i, "{}".format(i)])

    return render_template('index.html', data=model_list, bar_data={"c": bar_data_color_l, "d": bar_data_depth_l, "t": bar_ticks})

if __name__ == '__main__':
    app.run(host= '0.0.0.0')
