import numpy as np
import pandas as pd

import tagnews


class Test_GeoCoder:
    @classmethod
    def setup_class(cls):
        cls.model = tagnews.GeoCoder()

    def test_extract_geostrings(self):
        self.model.extract_geostrings(
            (
                "This is example article text with a location of"
                " 55th and Woodlawn where something happened."
            )
        )

    def test_extract_geostring_probs(self):
        article = (
            "This is example article text with a location of"
            " 55th and Woodlawn where something happened."
        )
        words, probs = self.model.extract_geostring_probs(article)
        max_prob = probs.max()
        max_word = words[np.argmax(probs)]
        geostrings = self.model.extract_geostrings(article,
                                                   prob_thresh=max_prob-0.001)
        assert max_word in [word for geostring in geostrings for word in geostring][0]

    def test_extract_geostring_probs_word_not_in_glove(self):
        """
        Regression test for issue #105.
        """
        article = "___1234567890nonexistent0987654321___"
        words, probs = self.model.extract_geostring_probs(article)

    def test_lat_longs_from_geostring_lists(self):
        geostring_lists = [
            ["5500", "S", "Woodlawn"],
            ["100", "N.", "Wacker"],
            ["thigh"],
        ]
        coords, scores = self.model.lat_longs_from_geostring_lists(
            geostring_lists, sleep_secs=0.0
        )

        assert coords.shape[0] == len(geostring_lists) == len(scores)

    def test_community_areas(self):
        # Approximately 55th and Woodlawn, which is in Hyde Park.
        coords = pd.DataFrame([[41.793465, -87.596930]], columns=["lat", "long"])
        com_area = self.model.community_area_from_coords(coords)
        assert com_area == ["HYDE PARK"]

    def test_best_geostring(self):
        """Verify that the best_geostring function returns expected values"""
        # Example from the readme
        input1 = (
            [
                ["1700", "block", "of", "S.", "Halsted", "Ave."],
                ["55th", "and", "Woodlawn,"],
            ],
            [
                np.array(
                    [
                        0.71738559,
                        0.81395197,
                        0.82227415,
                        0.79400611,
                        0.70529455,
                        0.60538059,
                    ]
                ),
                np.array(
                    [
                        0.79358339,
                        0.69696939,
                        0.68011874
                    ]
                ),
            ],
        )
        output1 = ["1700", "block", "of", "S.", "Halsted", "Ave."]
        # Empty geostring example
        input2, output2 = [(), ()], ''
        for inpt, expected_output in zip([input1, input2], [output1, output2]):
            actual_output = self.model.best_geostring(inpt)
            assert (
                actual_output == expected_output
            ), "ERROR: expected output != actual output for input {}/n  {} != {}".format(
                inpt, actual_output, expected_output
            )
