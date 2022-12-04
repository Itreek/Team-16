from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import inspect
import logging
import time

import numpy as np
import pandas as pd

from math_util import quiet_divide
from lap import linear_sum_assignment
from mot import MOTAccumulator

try:
    _getargspec = inspect.getfullargspec
except AttributeError:
    _getargspec = inspect.getargspec


class MetricsHost:
    def __init__(self):
        self.metrics = OrderedDict()
    def register(
        self,
        fnc,
        deps="auto",
        name=None,
        helpstr=None,
        formatter=None,
        fnc_m=None,
        deps_m="auto",
    ):
        

        if deps is None:
            deps = []
        elif deps == "auto":
            if _getargspec(fnc).defaults is not None:
                k = -len(_getargspec(fnc).defaults)
            else:
                k = len(_getargspec(fnc).args)
            deps = _getargspec(fnc).args[1:k]  # assumes dataframe as first argument

        if name is None:
            name = (
                fnc.__name__
            )  # Relies on meaningful function names, i.e don't use for lambdas

        if helpstr is None:
            helpstr = inspect.getdoc(fnc) if inspect.getdoc(fnc) else "No description."
            helpstr = " ".join(helpstr.split())
        if fnc_m is None and name + "_m" in globals():
            fnc_m = globals()[name + "_m"]
        if fnc_m is not None:
            if deps_m is None:
                deps_m = []
            elif deps_m == "auto":
                if _getargspec(fnc_m).defaults is not None:
                    k = -len(_getargspec(fnc_m).defaults)
                else:
                    k = len(_getargspec(fnc_m).args)
                deps_m = _getargspec(fnc_m).args[
                    1:k
                ]  # assumes dataframe as first argument
        else:
            deps_m = None

        self.metrics[name] = {
            "name": name,
            "fnc": fnc,
            "fnc_m": fnc_m,
            "deps": deps,
            "deps_m": deps_m,
            "help": helpstr,
            "formatter": formatter,
        }

    @property
    def names(self):
        """Returns the name identifiers of all registered metrics."""
        return [v["name"] for v in self.metrics.values()]

    @property
    def formatters(self):
        """Returns the formatters for all metrics that have associated formatters."""
        return {
            k: v["formatter"]
            for k, v in self.metrics.items()
            if v["formatter"] is not None
        }

    def list_metrics(self, include_deps=False):
        """Returns a dataframe containing names, descriptions and optionally dependencies for each metric."""
        cols = ["Name", "Description", "Dependencies"]
        if include_deps:
            data = [(m["name"], m["help"], m["deps"]) for m in self.metrics.values()]
        else:
            data = [(m["name"], m["help"]) for m in self.metrics.values()]
            cols = cols[:-1]

        return pd.DataFrame(data, columns=cols)

    def list_metrics_markdown(self, include_deps=False):
        """Returns a markdown ready version of `list_metrics`."""
        df = self.list_metrics(include_deps=include_deps)
        fmt = [":---" for i in range(len(df.columns))]
        df_fmt = pd.DataFrame([fmt], columns=df.columns)
        df_formatted = pd.concat([df_fmt, df])
        return df_formatted.to_csv(sep="|", index=False)

    def compute(
        self,
        df,
        ana=None,
        metrics=None,
        return_dataframe=True,
        return_cached=False,
        name=None,
    ):
        
        if isinstance(df, MOTAccumulator):
            df = df.events

        if metrics is None:
            metrics = motchallenge_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]

        df_map = events_to_df_map(df)

        cache = {}
        options = {"ana": ana}
        for mname in metrics:
            cache[mname] = self._compute(
                df_map, mname, cache, options, parent="summarize"
            )

        if name is None:
            name = 0

        if return_cached:
            data = cache
        else:
            data = OrderedDict([(k, cache[k]) for k in metrics])

        ret = pd.DataFrame(data, index=[name]) if return_dataframe else data
        return ret

    def compute_overall(
        self,
        partials,
        metrics=None,
        return_dataframe=True,
        return_cached=False,
        name=None,
    ):
       
        if metrics is None:
            metrics = motchallenge_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]
        cache = {}

        for mname in metrics:
            cache[mname] = self._compute_overall(
                partials, mname, cache, parent="summarize"
            )

        if name is None:
            name = 0
        if return_cached:
            data = cache
        else:
            data = OrderedDict([(k, cache[k]) for k in metrics])
        return pd.DataFrame(data, index=[name]) if return_dataframe else data

       
simple_add_func = []


def num_frames(df):
    """Total number of frames."""
    return df.full.index.get_level_values(0).unique().shape[0]


simple_add_func.append(num_frames)


def obj_frequencies(df):
    """Total number of occurrences of individual objects over all frames."""
    return df.noraw.OId.value_counts()


def pred_frequencies(df):
    """Total number of occurrences of individual predictions over all frames."""
    return df.noraw.HId.value_counts()



def num_matches(df):
    """Total number matches."""
    return df.noraw.Type.isin(["MATCH"]).sum()


simple_add_func.append(num_matches)


def num_false_positives(df):
    """Total number of false positives (false-alarms)."""
    return df.noraw.Type.isin(["FP"]).sum()


simple_add_func.append(num_false_positives)


def num_misses(df):
    """Total number of misses."""
    return df.noraw.Type.isin(["MISS"]).sum()


simple_add_func.append(num_misses)


def num_detections(df, num_matches, num_switches):
    """Total number of detected objects including matches and switches."""
    del df  # unused
    return num_matches + num_switches


simple_add_func.append(num_detections)

def num_fragmentations(df, obj_frequencies):
    """Total number of switches from tracked to not tracked."""
    fra = 0
    for o in obj_frequencies.index:
        # Find first and last time object was not missed (track span). Then count
        # the number switches from NOT MISS to MISS state.
        dfo = df.noraw[df.noraw.OId == o]
        notmiss = dfo[dfo.Type != "MISS"]
        if len(notmiss) == 0:
            continue
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == "MISS" else 0).diff()
        fra += diffs[diffs == 1].count()
    return fra


simple_add_func.append(num_fragmentations)


def motp(df, num_detections):
    """Multiple object tracker precision."""
    return quiet_divide(df.noraw["D"].sum(), num_detections)



def mota(df, num_misses, num_switches, num_false_positives, num_objects):
    """Multiple object tracker accuracy."""
    del df  # unused
    return 1.0 - quiet_divide(
        num_misses + num_switches + num_false_positives, num_objects
    )



def precision(df, num_detections, num_false_positives):
    """Number of detected objects over sum of detected and false positives."""
    del df  # unused
    return quiet_divide(num_detections, num_false_positives + num_detections)


def recall(df, num_detections, num_objects):
    """Number of detections over number of objects."""
    del df  # unused
    return quiet_divide(num_detections, num_objects)



class DataFrameMap:  # pylint: disable=too-few-public-methods
    def __init__(self, full, raw, noraw, extra):
        self.full = full
        self.raw = raw
        self.noraw = noraw
        self.extra = extra


def events_to_df_map(df):
    raw = df[df.Type == "RAW"]
    noraw = df[
        (df.Type != "RAW")
        & (df.Type != "ASCEND")
        & (df.Type != "TRANSFER")
        & (df.Type != "MIGRATE")
    ]
    extra = df[df.Type != "RAW"]
    df_map = DataFrameMap(full=df, raw=raw, noraw=noraw, extra=extra)
    return df_map


def extract_counts_from_df_map(df):
    """
    Returns:
        Tuple (ocs, hcs, tps).
        ocs: Dict from object id to count.
        hcs: Dict from hypothesis id to count.
        tps: Dict from (object id, hypothesis id) to true-positive count.
        The ids are arbitrary, they might NOT be consecutive integers from 0.
    """
    oids = df.full["OId"].dropna().unique()
    hids = df.full["HId"].dropna().unique()

    flat = df.raw.reset_index()
    # Exclude events that do not belong to either set.
    flat = flat[flat["OId"].isin(oids) | flat["HId"].isin(hids)]
    # Count number of frames where each (non-empty) OId and HId appears.
    ocs = flat.set_index("OId")["FrameId"].groupby("OId").nunique().to_dict()
    hcs = flat.set_index("HId")["FrameId"].groupby("HId").nunique().to_dict()
    # Select three columns of interest and index by ('OId', 'HId').
    dists = flat[["OId", "HId", "D"]].set_index(["OId", "HId"]).dropna()
    # Count events with non-empty distance for each pair.
    tps = dists.groupby(["OId", "HId"])["D"].count().to_dict()
    return ocs, hcs, tps


def id_global_assignment(df, ana=None):
    """ID measures: Global min-cost assignment for ID measures."""
    # pylint: disable=too-many-locals
    del ana  # unused
    ocs, hcs, tps = extract_counts_from_df_map(df)
    oids = sorted(ocs.keys())
    hids = sorted(hcs.keys())
    oids_idx = dict((o, i) for i, o in enumerate(oids))
    hids_idx = dict((h, i) for i, h in enumerate(hids))
    no = len(ocs)
    nh = len(hcs)

    fpmatrix = np.full((no + nh, no + nh), 0.0)
    fnmatrix = np.full((no + nh, no + nh), 0.0)
    fpmatrix[no:, :nh] = np.nan
    fnmatrix[:no, nh:] = np.nan

    for oid, oc in ocs.items():
        r = oids_idx[oid]
        fnmatrix[r, :nh] = oc
        fnmatrix[r, nh + r] = oc

    for hid, hc in hcs.items():
        c = hids_idx[hid]
        fpmatrix[:no, c] = hc
        fpmatrix[c + no, c] = hc

    for (oid, hid), ex in tps.items():
        r = oids_idx[oid]
        c = hids_idx[hid]
        fpmatrix[r, c] -= ex
        fnmatrix[r, c] -= ex

    costs = fpmatrix + fnmatrix
    rids, cids = linear_sum_assignment(costs)

    return {
        "fpmatrix": fpmatrix,
        "fnmatrix": fnmatrix,
        "rids": rids,
        "cids": cids,
        "costs": costs,
        "min_cost": costs[rids, cids].sum(),
    }


def idf1(df, idtp, num_objects, num_predictions):
    """ID measures: global min-cost F1 score."""
    del df  # unused
    return quiet_divide(2 * idtp, num_objects + num_predictions)


for one in simple_add_func:
    name = one.__name__

    def getSimpleAdd(nm):
        def simpleAddHolder(partials):
            res = 0
            for v in partials:
                res += v[nm]
            return res

        return simpleAddHolder

    locals()[name + "_m"] = getSimpleAdd(name)


def create():
    """Creates a MetricsHost and populates it with default metrics."""
    m = MetricsHost()

    m.register(num_frames, formatter="{:d}".format)
    m.register(obj_frequencies, formatter="{:d}".format)
    m.register(pred_frequencies, formatter="{:d}".format)
    m.register(num_matches, formatter="{:d}".format)
    m.register(num_misses, formatter="{:d}".format)
    m.register(num_detections, formatter="{:d}".format)
    m.register(motp, formatter="{:.3f}".format)
    m.register(mota, formatter="{:.1%}".format)
    m.register(precision, formatter="{:.1%}".format)
    m.register(recall, formatter="{:.1%}".format)
    m.register(idf1, formatter="{:.1%}".format)

    return m


motchallenge_metrics = [
    "idf1",
    "recall",
    "precision",
    "num_unique_objects",
    "num_misses",
    "mota",
    "motp",
]


