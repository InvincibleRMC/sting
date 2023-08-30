import os
import random
from unittest import TestCase

import numpy as np
import pandas as pd

from sting.data import (
    Feature,
    FeatureType,
    _find_file,
    _parse_and_preprocess_csv,
    _parse_schema,
    _trim_line,
    parse_c45,
)


# noinspection DuplicatedCode
class TestFeature(TestCase):
    def test_string_ftype(self):
        # using the string name for each feature type should work and be casted
        f = Feature("f", "BINARY")
        self.assertTrue(f.ftype == FeatureType.BINARY)
        f = Feature("f", "NOMINAL", ["a", "b", "c"])
        self.assertTrue(f.ftype == FeatureType.NOMINAL)
        f = Feature("f", "CONTINUOUS")
        self.assertTrue(f.ftype == FeatureType.CONTINUOUS)

    def test_values_attribute(self):
        # Failing to define the values attribute for a NOMINAL feature should cause a ValueError
        self.assertRaises(ValueError, lambda: Feature("f", FeatureType.NOMINAL))
        # Defining the values attribute for any other feature type should cause a ValueError
        self.assertRaises(
            ValueError, lambda: Feature("f", FeatureType.BINARY, ["a", "b", "c"])
        )
        self.assertRaises(
            ValueError, lambda: Feature("f", FeatureType.CONTINUOUS, ["a", "b", "c"])
        )

    def test_order(self):
        # Assert that Feature instances are orderable
        # noinspection PyTypeChecker
        sorted(
            [
                Feature("f_binary", FeatureType.BINARY),
                Feature("f_nominal", FeatureType.NOMINAL, ["a", "b", "c"]),
                Feature("f_cont", FeatureType.CONTINUOUS),
            ]
        )

    def test_to_float(self):
        v = ["a", "b", "c"]

        f = Feature("f", "BINARY")
        # binary features should convert to their equivalent float value
        self.assertAlmostEqual(0.0, f.to_float(0))
        self.assertAlmostEqual(1.0, f.to_float(1))
        self.assertAlmostEqual(0.0, f.to_float(False))
        self.assertAlmostEqual(1.0, f.to_float(True))

        # Nominal features should convert to their associated enum value
        f = Feature("f", "NOMINAL", v)
        self.assertAlmostEqual(1.0, f.to_float(f.values.a))
        self.assertAlmostEqual(1.0, f.to_float("a"))
        self.assertAlmostEqual(1.0, f.to_float(1))
        self.assertAlmostEqual(2.0, f.to_float(f.values.b))
        self.assertAlmostEqual(2.0, f.to_float("b"))
        self.assertAlmostEqual(2.0, f.to_float(2))
        self.assertAlmostEqual(3.0, f.to_float(f.values.c))
        self.assertAlmostEqual(3.0, f.to_float("c"))
        self.assertAlmostEqual(3.0, f.to_float(3))

        # Continuous features should be the same because they are already floats
        f = Feature("f", "CONTINUOUS")
        for _ in range(100):
            n = random.random()
            self.assertAlmostEqual(n, f.to_float(n))

    def test_from_float(self):
        v = ["a", "b", "c"]

        # Binary should convert to True and False
        f = Feature("f", "BINARY")
        self.assertEqual(False, f.from_float(0.0))
        self.assertEqual(True, f.to_float(1.0))

        # Nominal features should convert to their original enum values
        f = Feature("f", "NOMINAL", v)
        self.assertEqual(f.values.a, f.from_float(1.0))
        self.assertEqual(f.values.b, f.from_float(2.0))
        self.assertEqual(f.values.c, f.from_float(3.0))

        # Continuous should stay the same because they're just floats
        f = Feature("f", "CONTINUOUS")
        for _ in range(100):
            n = random.random()
            self.assertAlmostEqual(n, f.from_float(n))


SCHEMA = [
    Feature("f1", FeatureType.BINARY),
    Feature(
        "f2",
        FeatureType.NOMINAL,
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    ),
    Feature("f3", FeatureType.CONTINUOUS),
    Feature(
        "f4", FeatureType.NOMINAL, ["A12", "A13", "A14"]
    ),
    Feature(
        "f5", FeatureType.NOMINAL, ["1", "2", "3"]
    ),
]

DATA = [
    [0.0, 1.0, 0.94, 1.0, 1.0],
    [0.0, 2.0, 1.0, 2.0, 2.0],
    [0.0, 3.0, 1.5, 3.0, 3.0],
    [np.nan, 4.0, 11e-1, 3.0, 2.0],
    [0.0, 5.0, 2.3, 1.0, 1.0],
    [1.0, 3.0, 0.86, 2.0, 2.0],
    [0.0, 2.0, 3.14, 1.0, 3.0],
    [0.0, 1.0, 2.81, np.nan, 2.0],
    [1.0, 4.0, 0.9932456, 2.0, 1.0],
    [1.0, 5.0, 2.0, 1.0, 2.0],
]

LABELS = [0, 0, 0, 1, 1, 0, 1, 0, 1, 1]

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "example_data")


class TestParse(TestCase):
    def test_find_file(self):
        self.assertEqual(
            os.path.join(DATA_DIR, "example.data"), _find_file("example.data", ROOT_DIR)
        )
        self.assertEqual(
            os.path.join(DATA_DIR, "example.names"),
            _find_file("example.names", ROOT_DIR),
        )

    def test_trim_line(self):
        self.assertEqual("", _trim_line(""))
        self.assertEqual("", _trim_line("#"))

        LINE = "6,1,Wednesday,0.86,A13,0. # lines may end with periods"
        self.assertEqual("6,1,Wednesday,0.86,A13,0", _trim_line(LINE))

    def test_parse_csv(self):
        expected_schema = SCHEMA
        df = _parse_and_preprocess_csv(
            expected_schema, os.path.join(DATA_DIR, "example.data")
        )
        df.pop("label")

        expected_df = pd.DataFrame(DATA, columns=["f1", "f2", "f3", "f4", "f5"])
        expected_df["index"] = np.arange(1, len(expected_df) + 1)
        expected_df.set_index("index", inplace=True)

        pd.testing.assert_frame_equal(expected_df, df)

    def test_parse_schema(self):
        expected_schema = SCHEMA
        schema = _parse_schema(os.path.join(DATA_DIR, "example.names"))
        self.assertEqual(expected_schema, schema)

    def test_parse_c45(self):
        schema, X, y = parse_c45("example", DATA_DIR)

        self.assertEqual(SCHEMA, schema)
        np.testing.assert_array_almost_equal(X, np.array(DATA, dtype=float))
        np.testing.assert_array_equal(y, np.array(LABELS, dtype=int))
