from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import itertools

import numpy as np
import pandas as pd
from lap import linear_sum_assignment

_INDEX_FIELDS = ['FrameId', 'Event']
_EVENT_FIELDS = ['Type', 'OId', 'HId', 'D']

class MOTAccumulator(object):
    def __init__(self, auto_id=False, max_switch_time=float('inf')):
       
        # Parameters of the accumulator.
        self.auto_id = auto_id
        self.max_switch_time = max_switch_time

        # Accumulator state.
        self._events = None
        self._indices = None
        self.m = None
        self.res_m = None
        self.last_occurrence = None
        self.last_match = None
        self.hypHistory = None
        self.dirty_events = None
        self.cached_events_df = None

        self.reset()

    def reset(self):
        """Reset the accumulator to empty state."""

        self._events = {field: [] for field in _EVENT_FIELDS}
        self._indices = {field: [] for field in _INDEX_FIELDS}
        self.m = {}  # Pairings up to current timestamp
        self.res_m = {}  # Result pairings up to now
        self.last_occurrence = {}  # Tracks most recent occurance of object
        self.last_match = {}  # Tracks most recent match of object
        self.hypHistory = {}
        self.dirty_events = True
        self.cached_events_df = None

    def _append_to_indices(self, frameid, eid):
        self._indices['FrameId'].append(frameid)
        self._indices['Event'].append(eid)

    def _append_to_events(self, typestr, oid, hid, distance):
        self._events['Type'].append(typestr)
        self._events['OId'].append(oid)
        self._events['HId'].append(hid)
        self._events['D'].append(distance)

    def update(self, oids, hids, dists, frameid = None, vf=''):        
        self.dirty_events = True
        oids = np.asarray(oids[0])
        oids_masked = np.zeros_like(oids, dtype=bool)
        hids = np.asarray(hids[0])
        hids_masked = np.zeros_like(hids, dtype=bool)

        if frameid is None:
            assert self.auto_id, 'auto-id is not enabled'
            if len(self._indices['FrameId']) > 0:
                frameid = self._indices['FrameId'][-1] + 1
            else:
                frameid = 0
        else:
            assert not self.auto_id, 'Cannot provide frame id when auto-id is enabled'

        eid = itertools.count()

        # 0. Record raw events

        no = len(oids)
        nh = len(hids)
        print(no, nh)
        # Add a RAW event simply to ensure the frame is counted.
        self._append_to_indices(frameid, next(eid))
        self._append_to_events('RAW', np.nan, np.nan, np.nan)
        # print(oids)
        # There must be at least one RAW event per object and hypothesis.
        # Record all finite distances as RAW events.

        print(len(dists), len(dists[0]))
        valid_i, valid_j = np.where(np.isfinite(dists))
        print(valid_i, valid_j)
        valid_dists = dists[valid_i, valid_j]
        for i, j, dist_ij in zip(valid_i, valid_j, valid_dists):
            self._append_to_indices(frameid, next(eid))
            # print(i, j, dist_ij)
            self._append_to_events('RAW', oids[i], hids[j], dist_ij)
        # Add a RAW event for objects and hypotheses that were present but did
        # not overlap with anything.
        used_i = np.unique(valid_i)
        used_j = np.unique(valid_j)
        unused_i = np.setdiff1d(np.arange(no), used_i)
        unused_j = np.setdiff1d(np.arange(nh), used_j)
        for oid in oids[unused_i]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('RAW', oid, np.nan, np.nan)
        for hid in hids[unused_j]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('RAW', np.nan, hid, np.nan)

        if oids.size * hids.size > 0:
            # 1. Try to re-establish tracks from previous correspondences
            for i in range(oids.shape[0]):
                # No need to check oids_masked[i] here.
                if oids[i][0] not in self.m:
                    continue

                hprev = self.m[oids[i]]
                j, = np.where(~hids_masked & (hids == hprev))
                if j.shape[0] == 0:
                    continue
                j = j[0]

                if np.isfinite(dists[i, j]):
                    o = oids[i]
                    h = hids[j]
                    oids_masked[i] = True
                    hids_masked[j] = True
                    self.m[oids[i]] = hids[j]

                    self._append_to_indices(frameid, next(eid))
                    self._append_to_events('MATCH', oids[i], hids[j], dists[i, j])
                    self.last_match[o] = frameid
                    self.hypHistory[h] = frameid

            # 2. Try to remaining objects/hypotheses
            print(oids_masked)
            dists[oids_masked, :] = np.nan
            dists[:, hids_masked] = np.nan

            rids, cids = linear_sum_assignment(dists)

            for i, j in zip(rids, cids):
                if not np.isfinite(dists[i, j]):
                    continue

                o = oids[i]
                h = hids[j]
                is_switch = (o in self.m and
                             self.m[o] != h and
                             abs(frameid - self.last_occurrence[o]) <= self.max_switch_time)
                cat1 = 'SWITCH' if is_switch else 'MATCH'
                if cat1 == 'SWITCH':
                    if h not in self.hypHistory:
                        subcat = 'ASCEND'
                        self._append_to_indices(frameid, next(eid))
                        self._append_to_events(subcat, oids[i], hids[j], dists[i, j])
                # ignore the last condition temporarily
                is_transfer = (h in self.res_m and
                               self.res_m[h] != o)
                # is_transfer = (h in self.res_m and
                #                self.res_m[h] != o and
                #                abs(frameid - self.last_occurrence[o]) <= self.max_switch_time)
                cat2 = 'TRANSFER' if is_transfer else 'MATCH'
                if cat2 == 'TRANSFER':
                    if o not in self.last_match:
                        subcat = 'MIGRATE'
                        self._append_to_indices(frameid, next(eid))
                        self._append_to_events(subcat, oids[i], hids[j], dists[i, j])
                    self._append_to_indices(frameid, next(eid))
                    self._append_to_events(cat2, oids[i], hids[j], dists[i, j])
                if vf != '' and (cat1 != 'MATCH' or cat2 != 'MATCH'):
                    if cat1 == 'SWITCH':
                        vf.write('%s %d %d %d %d %d\n' % (subcat[:2], o, self.last_match[o], self.m[o], frameid, h))
                    if cat2 == 'TRANSFER':
                        vf.write('%s %d %d %d %d %d\n' % (subcat[:2], h, self.hypHistory[h], self.res_m[h], frameid, o))
                self.hypHistory[h] = frameid
                self.last_match[o] = frameid
                self._append_to_indices(frameid, next(eid))
                self._append_to_events(cat1, oids[i], hids[j], dists[i, j])
                oids_masked[i] = True
                hids_masked[j] = True
                self.m[o] = h
                self.res_m[h] = o

        # 3. All remaining objects are missed
        for o in oids[~oids_masked]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('MISS', o, np.nan, np.nan)
            if vf != '':
                vf.write('FN %d %d\n' % (frameid, o))

        # 4. All remaining hypotheses are false alarms
        for h in hids[~hids_masked]:
            self._append_to_indices(frameid, next(eid))
            self._append_to_events('FP', np.nan, h, np.nan)
            if vf != '':
                vf.write('FP %d %d\n' % (frameid, h))

        # 5. Update occurance state
        for o in oids:
            self.last_occurrence[o] = frameid

        return frameid

    @property
    def events(self):
        if self.dirty_events:
            self.cached_events_df = MOTAccumulator.new_event_dataframe_with_data(self._indices, self._events)
            self.dirty_events = False
        return self.cached_events_df

    @property
    def mot_events(self):
        df = self.events
        return df[df.Type != 'RAW']

    @staticmethod
    def new_event_dataframe():
        """Create a new DataFrame for event tracking."""
        idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['FrameId', 'Event'])
        cats = pd.Categorical([], categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'])
        df = pd.DataFrame(
            OrderedDict([
                ('Type', pd.Series(cats)),          # Type of event. One of FP (false positive), MISS, SWITCH, MATCH
                ('OId', pd.Series(dtype=float)),      # Object ID or -1 if FP. Using float as missing values will be converted to NaN anyways.
                ('HId', pd.Series(dtype=float)),      # Hypothesis ID or NaN if MISS. Using float as missing values will be converted to NaN anyways.
                ('D', pd.Series(dtype=float)),      # Distance or NaN when FP or MISS
            ]),
            index=idx
        )
        return df

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        
        if len(events) == 0:
            return MOTAccumulator.new_event_dataframe()

        raw_type = pd.Categorical(
            events['Type'],
            categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND', 'MIGRATE'],
            ordered=False)
        series = [
            pd.Series(raw_type, name='Type'),
            pd.Series(events['OId'], dtype=float, name='OId'),
            pd.Series(events['HId'], dtype=float, name='HId'),
            pd.Series(events['D'], dtype=float, name='D')
        ]

        idx = pd.MultiIndex.from_arrays(
            [indices[field] for field in _INDEX_FIELDS],
            names=_INDEX_FIELDS)
        df = pd.concat(series, axis=1)
        df.index = idx
        return df

    @staticmethod
    def merge_analysis(anas, infomap):
        # pylint: disable=missing-function-docstring
        res = {'hyp': {}, 'obj': {}}
        mapp = {'hyp': 'hid_map', 'obj': 'oid_map'}
        for ana, infom in zip(anas, infomap):
            if ana is None:
                return None
            for t in ana.keys():
                which = mapp[t]
                if np.nan in infom[which]:
                    res[t][int(infom[which][np.nan])] = 0
                if 'nan' in infom[which]:
                    res[t][int(infom[which]['nan'])] = 0
                for _id, cnt in ana[t].items():
                    if _id not in infom[which]:
                        _id = str(_id)
                    res[t][int(infom[which][_id])] = cnt
        return res
    
    def merge_event_dataframes(dfs, update_frame_indices=True, update_oids=True, update_hids=True, return_mappings=False):
        
        mapping_infos = []
        new_oid = itertools.count()
        new_hid = itertools.count()

        r = MOTAccumulator.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulator):
                df = df.events

            copy = df.copy()
            infos = {}

            # Update index
            if update_frame_indices:
                # pylint: disable=cell-var-from-loop
                next_frame_id = max(r.index.get_level_values(0).max() + 1, r.index.get_level_values(0).unique().shape[0])
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:
                # pylint: disable=cell-var-from-loop
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map

            if update_hids:
                # pylint: disable=cell-var-from-loop
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map

            r = r.append(copy)
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:
            return r

