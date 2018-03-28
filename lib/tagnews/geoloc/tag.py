from __future__ import division

import os
from collections import namedtuple
import glob
import time

import geocoder
import pandas as pd
import numpy as np
import re

from .. import utils

from contextlib import ExitStack, redirect_stderr

with ExitStack() as stack:
    null_stream = open(os.devnull, "w")
    stack.enter_context(null_stream)
    stack.enter_context(redirect_stderr(null_stream))
    import keras

"""
Contains the CrimeTags class that allows tagging of articles.
"""

MODEL_LOCATION = os.path.join(os.path.split(__file__)[0],
                              os.path.join('models', 'lstm', 'saved'))


def post_process(geostring):
    """
    Post process the geostring in a way that makes it more amenable to
    geocoding by the current geocoding provider GISgraphy.

    Inputs
    ------
    geostring : str
        The geostring to post process

    Returns
    -------
    processed_geostring : str
    """
    # Merge multiple whitespaces into one
    geostring = ' '.join(geostring.split())

    # add chicago to the end if it's not already in there and 'illinois'
    # is not in there. If 'illinois' is in there then there's a good
    # chance the city name is already in there.
    if 'chicago' not in geostring.lower() and 'illinois' not in geostring.lower():
        geostring = geostring + ' Chicago'

    # add illinois to the end if it's not already in there
    if ('illinois' not in geostring.lower()
            and not geostring.lower().endswith(' il')
            and ' il ' not in geostring.lower()):
        geostring = geostring + ' Illinois'

    # gisgraphy struggles with things like "55th and Woodlawn".
    # replace "...<number><number ender, e.g. th or rd> and..."
    # with two zeros.
    # \100 does not work correclty so we need to add a separator.
    geostring = re.sub(r'([0-9]+)[th|rd|st] and',
                       r'\1<__internal_separator__>00 and',
                       geostring)
    geostring = geostring.replace('<__internal_separator__>', '')

    # remove stopwords, only if they are internal, i.e.
    # the geostring doesn't start with "block ...".
    for stopword in ['block', 'of', 'and']:
        geostring = geostring.replace(' {} '.format(stopword), ' ')

    return geostring


GeocodeResults = namedtuple('GeocodeResults', ['lat_longs_raw',
                                               'full_responses_raw',
                                               'scores_raw',
                                               'lat_longs_post',
                                               'full_responses_post',
                                               'scores_post'])


def get_lat_longs_from_geostrings(geostring_list, post_process_f=None, sleep_secs=0):
    """
    Geo-code each geostring in `geostring_list` into lat/long values.
    Also return the full response from the geocoding service.

    Inputs
    ------
    geostring_list : list of strings
        The list of geostrings to geocode into lat/longs.
    post_process_f : function
        The results are returned for both the raw geostrings being
        passed to the geocoder, and the results of
        `post_process_f(geostring)` being passed to the geocoder.
    sleep_secs : float
        How long to sleep between successive requests, in seconds.

    Returns
    -------
    GeocodeResults : namedtuple
        A named tuple with the following fields:
        lat_longs_raw : list of tuples
            The length `n` list of lat/long tuple pairs or None.
        full_responses_raw : list
            The length `n` list of the full responses from the geocoding
            service.
        lat_longs_post : list of tuples
            The length `n` list of lat/long tuple pairs or None of the
            post-processed geostrings.
        full_responses_post : list
            The length `n` list of the full responses of the post-processed
            geostrings.
    """
    if post_process_f is None:
        post_process_f = post_process

    def _geocode(lst):
        full_responses = []
        for addr_str in geostring_list:
            g = geocoder.gisgraphy(addr_str)
            full_responses.append(g)
            time.sleep(sleep_secs)

        lat_longs = [g.latlng for g in full_responses]

        scores = []
        for g in full_responses:
            try:
                scores.append(json.loads(g.response.content)['result'][0]['score'])
            except Exception:
                scores.append(float('nan'))
        scores = np.array(scores, dtype='float32')

        return full_responses, lat_longs, scores

    full_responses_raw, lat_longs_raw, scores_raw = _geocode(geostring_list)

    geostring_list = [post_process_f(geo_s) for geo_s in geostring_list]
    full_responses_post, lat_longs_post, scores_post = _geocode(geostring_list)

    return GeocodeResults(lat_longs_raw=lat_longs_raw,
                          full_responses_raw=full_responses_raw,
                          scores_raw=scores_raw,
                          lat_longs_post=lat_longs_post,
                          full_responses_post=full_responses_post,
                          scores_post=scores_post)


def load_model(location=MODEL_LOCATION):
    """
    Load a model from the given folder `location`.
    There should be at least one file named model-TIME.pkl and
    a file named vectorizer-TIME.pkl inside the folder.

    The files with the most recent timestamp are loaded.
    """
    models = glob.glob(os.path.join(location, 'weights*.hdf5'))
    if not models:
        raise RuntimeError(('No models to load. Run'
                            ' "python -m tagnews.geoloc.models.'
                            'lstm.save_model"'))

    model = keras.models.load_model(models[-1])

    return model


class GeoCoder():
    def __init__(self):
        self.model = load_model()
        self.glove = utils.load_vectorizer.load_glove(
            os.path.join(os.path.split(__file__)[0],
                         '../data/glove.6B.50d.txt')
        )

    def pre_process(self, s):
        """
        Takes in a string which is the text of an article and returns the tuple
        `(words, data)` where `words` is the list of words found and `data`
        is the 3D numpy array that contains the numeric data that can be used
        by the trained model.

        Inputs
        ------
        s : str
            Article text.

        Returns
        -------
        words : list of strings
            The words found in the article.
        data : 3D numpy.array
            Has shape (1, N, M) where N is the number of words and M
            is the size of the word vectors, currently M is 51.
        """
        words = s.split() # split along white space.
        data = pd.concat([pd.DataFrame([[w[0].isupper()] if w else [False]
                                        for w in words]),
                          (self.glove.loc[words].fillna(0)
                           .reset_index(drop=True))],
                         axis='columns')
        return words, np.expand_dims(data, axis=0)

    def extract_geostring_probs(self, s):
        """
        Extract the probability that each word in s is part of a geostring.

        Inputs
        ------
        s : str
            Article text.

        Returns
        -------
        words : list of strings
            The words found in the article.
        probs : 1D numpy.array
            Has shape (N,) where N is the number of words.
        """
        if not s.strip():
            return [[], np.zeros((0,), dtype=np.float32)]
        words, data = self.pre_process(s)
        probs = self.model.predict(data)[0][:, 1]
        return words, probs

    def extract_geostrings(self, s, prob_thresh=0.5):
        """
        Extract the geostrings from the article text.

        Inputs
        ------
        s : str
            Article text.
        prob_thresh : float, 0 <= prob_thresh <= 1
            The threshold on probability above which words will be
            considered as part of a geostring.
            DEFAULT: 0.5

        Returns
        -------
        geostrings : list of lists of strings
            The list of extracted geostrings from the article text.
            Each word is kept separated in the list.
            Examle:
                [['1300', 'W.', 'Halsted'], ['Ohio']]
        """
        words, probs = self.extract_geostring_probs(s)
        above_thresh = probs >= prob_thresh

        words = ['filler'] + words + ['filler']
        above_thresh = np.concatenate([[False],
                                       above_thresh,
                                       [False]]).astype(np.int32)
        switch_ons = np.where(np.diff(above_thresh) == 1)[0] + 1
        switch_offs = np.where(np.diff(above_thresh) == -1)[0] + 1

        geostrings = []
        for on, off in zip(switch_ons, switch_offs):
            geostrings.append(words[on:off])

        return geostrings

    @staticmethod
    def lat_longs_from_geostring_lists(geostring_lists, **kwargs):
        """
        TODO
        """
        out = get_lat_longs_from_geostrings([' '.join(gl) for gl in geostring_lists],
                                            **kwargs)
        # the score returned from GISgraphy appears to be interpretable as
        # a higher score is LESS confident. Thus if raw / post > 1, then
        # we are less sure the location is correct/in chicago. Vice versa,
        # if  raw / post < 1, then we are more confident in the prediction.
        # Finally, if the raw couldn't geotag it but after post processing
        # we could, then we're more confident, so replace NaNs in the raw
        # with 0.
        out.scores_raw[np.isnan(out.scores_raw)] = 0
        # For all x >= 0, we have 0 <= 1 / (1 + x) <= 1, which is a nice
        # property to have.
        return out.lat_longs_post, 1 / (1 + out.scores_raw / out.scores_post)
