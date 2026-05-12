from batfit.utils.text_utils import shuffle_substrings


def test_no_delimiter_returns_single_item():
    result = shuffle_substrings("phi")
    assert result == ["phi"]


def test_two_parts_contains_both_orders():
    result = shuffle_substrings("phi-dvdq")
    assert "phi-dvdq" in result
    assert "dvdq-phi" in result


def test_two_parts_all_delimiters_present():
    result = shuffle_substrings("phi-dvdq")
    assert "phi_dvdq" in result
    assert "phi+dvdq" in result
    assert "dvdq_phi" in result


def test_result_contains_all_substrings():
    result = shuffle_substrings("phi-dvdq-dqdv")
    for item in result:
        assert "phi" in item
        assert "dvdq" in item
        assert "dqdv" in item


def test_underscore_delimiter_input():
    result = shuffle_substrings("a_b")
    assert "a-b" in result
    assert "a_b" in result
    assert "a+b" in result
    assert "b-a" in result


