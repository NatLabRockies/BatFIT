import os
import pickle
import tempfile

import numpy as np

from batfit.utils.data_utils import load_pickle


def test_load_pickle():
    obj = {"a": 1, "b": np.arange(3)}
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "obj.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        loaded = load_pickle(path)
    assert loaded["a"] == 1
    assert np.array_equal(loaded["b"], obj["b"])
