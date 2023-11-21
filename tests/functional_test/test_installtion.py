import pytest


def test_pkg_installation_1():
    try:
        import HousePricePrediction
    except Exception as e:
        assert False, f"Error{e} : House Price Prediction is not installed properly."
