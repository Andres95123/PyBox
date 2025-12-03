from pybox.example import add_one


def test_add_one():
    result = [add_one(test_input) for test_input in range(100)]
    assert result == [i+1 for i in range(100)]
