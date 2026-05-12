from batfit.utils.text_utils import shuffle_substrings


def test_shuffle_substrings():
    result = shuffle_substrings("phi")
    assert result == ["phi"]

    result = shuffle_substrings("phi-dvdq")
    assert "phi-dvdq" in result
    assert "dvdq-phi" in result

    result = shuffle_substrings("phi-dvdq")
    assert "phi_dvdq" in result
    assert "phi+dvdq" in result
    assert "dvdq_phi" in result


    result = shuffle_substrings("phi-dvdq-dqdv")
    for item in result:
        assert "phi" in item
        assert "dvdq" in item
        assert "dqdv" in item


    result = shuffle_substrings("a_b")
    assert "a-b" in result
    assert "a_b" in result
    assert "a+b" in result
    assert "b-a" in result


