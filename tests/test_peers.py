import unittest
import numpy as np
import pandas as pd
from src.peers import fund_returns


class PublishedNavTests(unittest.TestCase):
    def setUp(self):
        self.nav = pd.DataFrame({'code': ['000001'] * 4,
            'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-06']),
            'unit_nav': [10., 9., 9.9, 9.9], 'cash_per_unit': [0., 1., 0., 0.]})

    def test_dividend_reinvestment_before_calendar_alignment(self):
        # A 1-yuan distribution offsets 10 -> 9; growth on the extra publication
        # date is retained when the account calendar next observes the fund.
        cal = pd.DatetimeIndex(pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-06']))
        actual = fund_returns(self.nav, cal, ['000001'])
        np.testing.assert_allclose(actual['000001'], [0., 0., .1], atol=1e-15)

    def test_missing_common_date_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing'):
            fund_returns(self.nav, pd.DatetimeIndex(['2020-01-07']), ['000001'])

    def test_duplicate_publication_date_is_rejected(self):
        duplicated = pd.concat([self.nav, self.nav.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            fund_returns(duplicated, pd.DatetimeIndex(['2020-01-01']), ['000001'])
