import unittest
from src.slippage_math import adverse_price


class PriceSlippageTests(unittest.TestCase):
    def test_buy_three_basis_points_above_reference(self):
        self.assertAlmostEqual(adverse_price(10., 'buy', 3.), 10.003)

    def test_sell_three_basis_points_below_reference(self):
        self.assertAlmostEqual(adverse_price(10., 'sell', 3.), 9.997)

    def test_zero_slippage_preserves_reference_price(self):
        self.assertEqual(adverse_price(10., 'buy', 0.), 10.)
        self.assertEqual(adverse_price(10., 'sell', 0.), 10.)

    def test_round_trip_reduces_cash_even_without_price_change(self):
        buy = adverse_price(10., 'buy', 3.)
        sell = adverse_price(10., 'sell', 3.)
        self.assertAlmostEqual(1000 * (buy - sell), 6.)

    def test_bad_inputs_are_rejected(self):
        for reference, side, bps in [(0., 'buy', 3.), (10., 'buy', -1.),
                                     (10., 'sell', 10000.), (10., 'unknown', 3.)]:
            with self.assertRaises(ValueError):
                adverse_price(reference, side, bps)
