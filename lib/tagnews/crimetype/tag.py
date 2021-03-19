import os
import pickle
import glob
import time
import pandas as pd

# not used explicitly, but this needs to be imported like this
# for unpickling to work.
from ..utils.model_helpers import LemmaTokenizer # noqa

"""
Contains the CrimeTags class that allows tagging of articles.
"""

MODEL_LOCATION = os.path.join(os.path.split(__file__)[0],
                              'models',
                              'binary_stemmed_logistic')

TAGS = ['OEMC', 'CPD', 'SAO', 'CCCC', 'CCJ', 'CCSP',
        'CPUB', 'IDOC', 'DOMV', 'SEXA', 'POLB', 'POLM',
        'GUNV', 'GLBTQ', 'JUVE', 'REEN', 'VIOL', 'BEAT',
        'PROB', 'PARL', 'CPLY', 'DRUG', 'CPS', 'GANG', 'ILSP',
        'HOMI', 'IPRA', 'CPBD', 'IMMG', 'ENVI', 'UNSPC',
        'ILSC', 'ARSN', 'BURG', 'DUI', 'FRUD', 'ROBB', 'TASR']


def load_model(location=MODEL_LOCATION):
    """
    Load a model from the given folder `location`.
    There should be at least one file named model-TIME.pkl and
    a file named vectorizer-TIME.pkl inside the folder.

    The files with the most recent timestamp are loaded.
    """
    models = glob.glob(os.path.join(location, 'model*.pkl'))
    if not models:
        raise RuntimeError(('No models to load. Run'
                            ' "python -m tagnews.crimetype.models.'
                            'binary_stemmed_logistic.save_model"'))
    model = models.pop()
    while models:
        model_time = time.strptime(model[-19:-4], '%Y%m%d-%H%M%S')
        new_model_time = time.strptime(models[0][-19:-4], '%Y%m%d-%H%M%S')
        if model_time < new_model_time:
            model = models[0]
        models = models[1:]

    with open(model, 'rb') as f:
        clf = pickle.load(f)

    with open(os.path.join(location, 'vectorizer-' + model[-19:-4] + '.pkl'),
              'rb') as f:
        vectorizer = pickle.load(f)

    return clf, vectorizer


class CrimeTags():
    """
    CrimeTags let you tag articles. Neat!
    """
    def __init__(self,
                 model_directory=MODEL_LOCATION,
                 clf=None,
                 vectorizer=None):
        """
        Load a model from the given `model_directory`.
        See `load_model` for more information.

        Alternatively, the classifier and vectorizer can be
        provided. If one is provided, then both must be provided.
        """
        if clf is None and vectorizer is None:
            self.clf, self.vectorizer = load_model(model_directory)
        elif clf is None or vectorizer is None:
            raise ValueError(('clf and vectorizer must both be None,'
                              ' or both be not None'))
        else:
            self.clf, self.vectorizer = clf, vectorizer

    def tagtext_proba(self, text):
        """
        Compute the probability each tag applies to the given text.

        inputs:
            text: A python string.
        returns:
            pred_proba: A pandas series indexed by the tag name.
        """
        x = self.vectorizer.transform([text])
        y_hat = self.clf.predict_proba(x)
        preds = pd.DataFrame(y_hat)
        preds.columns = TAGS
        preds = preds.T.iloc[:, 0].sort_values(ascending=False)
        return preds

    def tagtext(self, text, prob_thresh=0.5):
        """
        Tag a string with labels.

        inputs:
            text: A python string.
            prob_thresh: The threshold on probability at which point
                the tag will be applied.
        returns:
            preds: A list of tags that have > prob_thresh probability
                according to the model.
        """
        preds = self.tagtext_proba(text)
        return preds[preds > prob_thresh].index.values.tolist()

    def relevant_proba(self, text):
        """
        Outputs the probability that the given text is relevant.
        This probability is computed naively as the maximum of
        the probabilities each tag applies to the text.

        A more nuanced method would compute a joint probability.

        inputs:
            text: A python string.

        returns:
            relevant_proba: Probability the text is relevant.
        """
        return max(self.tagtext_proba(text))

    def relevant(self, text, prob_thresh=0.05):
        """
        Determines whether given text is relevant or not. Relevance
        is defined as whether any tag has more than prob_thresh
        chance of applying to the text according to the model.

        inputs:
            text: A python string.
            prob_thresh: The threshold on probability that
                determines relevance. If no tags have >=
                prob_thresh of applying to the text, then
                the text is not relevant.
        returns:
            relevant: Boolean. Is the text "relevant"?
        """
        return len(self.tagtext(text, prob_thresh)) > 0

    def get_contributions(self, text):
        """
        Rank the words in the text by their contribution to each
        category. This function assumes that clf has an attribute
        `coef_` and that vectorizer has an attribute
        `inverse_transform`.

        inputs:
            text: A python string.
        returns:
            contributions: Pandas panel keyed off [category, word].

        Example:
        >>> s = 'This is an article about drugs and gangs.'
        >>> s += ' Written by the amazing Kevin Rose.'
        >>> p = tagger.get_contributions(s)
        >>> p['DRUG'].sort_values('weight', ascending=False)
                     weight
        drug       5.549870
        copyright  0.366905
        gang       0.194773
        this       0.124590
        an        -0.004484
        article   -0.052026
        is        -0.085534
        about     -0.154800
        kevin     -0.219028
        rose      -0.238296
        and       -0.316201
        .         -0.853208
        """
        vec = self.vectorizer.transform([text])
        vec_inv = self.vectorizer.inverse_transform(vec)[1]

        df = pd.DataFrame(columns=vec_inv, index=TAGS)

        for j, word in enumerate(vec_inv):
            for i, tag in enumerate(TAGS):
                df.loc[tag][word] = mod.clf.coef_[i, vec.nonzero()[1][j]]
        return df
