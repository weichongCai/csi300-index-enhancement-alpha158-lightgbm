"""Adverse execution prices; zero means the unchanged reference price."""
import math


def adverse_price(reference, side, bps):
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError('Reference price must be finite and positive.')
    if not math.isfinite(bps) or not 0 <= bps < 10000:
        raise ValueError('Slippage must be in [0, 10000) basis points.')
    if side not in ['buy', 'sell']:
        raise ValueError('Expected buy or sell.')
    return reference * (1 + bps/10000. if side == 'buy' else 1 - bps/10000.)
