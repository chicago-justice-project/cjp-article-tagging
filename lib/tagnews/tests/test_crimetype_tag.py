import tagnews
from tagnews.crimetype.tag import TAGS

class Test_Crimetype():
    @classmethod
    def setup_method(cls):
        cls.model = tagnews.CrimeTags()

    def test_tagtext(self):
        self.model.tagtext('This is example article text')

    def test_tagtext_proba(self):
        article = 'Murder afoul, someone has been shot!'
        probs = self.model.tagtext_proba(article)
        max_prob = probs.max()
        max_type = probs.idxmax()
        tags = self.model.tagtext(article,
                                  prob_thresh=max_prob - 0.001)
        assert max_type in tags

    def test_get_contributions(self):
        article = 'This is an article about drugs and gangs \
            written and photographed by the amazing spiderman'
        contributions = self.model.get_contributions(article)

        assert contributions.shape[0] == len(TAGS)
        assert contributions.shape[1] == (set(article.split(' ')))
