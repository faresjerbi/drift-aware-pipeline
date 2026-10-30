
from chatbot import is_refusal


def test_obvious_refusal():
    assert is_refusal("I cannot help with that.") is True


def test_alternative_refusal():
    assert is_refusal(
        "Sorry, I'm not able to provide that."
    ) is True  # Expected limitation of the current heuristic


def test_normal_response():
    assert is_refusal("Here is how to deploy your application.") is False


if __name__ == "__main__":
    test_obvious_refusal()
    test_alternative_refusal()
    test_normal_response()
    print("Refusal heuristic tests passed.")
