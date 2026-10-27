"""Construct the original Alpha158 expressions and load the correct frozen fold."""
from pathlib import Path
import numpy as np
import pandas as pd


def init_data(data_dir):
    import qlib
    data_dir = Path(data_dir).resolve()
    for name in ['features', 'calendars/day.txt', 'instruments/csi300.txt']:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f'Missing Qlib input: {name}')
    qlib.init(provider_uri=str(data_dir), region='cn', kernels=1,
              expression_cache=None, dataset_cache=None)
    return data_dir


def predict(root, start, end, *, instruments=None):
    import lightgbm as lgb
    from qlib.data import D
    from qlib.contrib.data.loader import Alpha158DL
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start < pd.Timestamp('2015-12-31') or end < start:
        raise ValueError('Prediction starts no earlier than 2015-12-31; end must follow start.')
    expressions, names = Alpha158DL.get_feature_config()
    # A small explicit subset is useful for deployment smoke checks; CLI defaults
    # to the historical CSI300 instrument filter.
    universe = D.instruments('csi300') if instruments is None else instruments
    features = D.features(universe, expressions,
                          start_time=str(start.date()), end_time=str(end.date()))
    if features.empty:
        raise ValueError('No historical members or quotes in the requested dates.')
    features.columns = names
    features = features.reorder_levels(['datetime', 'instrument']).sort_index()
    dates = features.index.get_level_values('datetime')
    scores = []
    # The later model is available only after its 2016-12-30 as-of close.
    for filename, mask in [
        ('first_2016_model.txt', dates < pd.Timestamp('2016-12-30')),
        ('frozen_2017_2025_model.txt', dates >= pd.Timestamp('2016-12-30')),
    ]:
        if not mask.any():
            continue
        model = lgb.Booster(model_file=str(Path(root) / 'models' / filename))
        if model.feature_name() != names:
            raise ValueError('Alpha158 feature names/order differ from the frozen model.')
        values = model.predict(features.loc[mask].to_numpy(), num_threads=4)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite model prediction.')
        scores.append(pd.Series(values, index=features.index[mask], name='score'))
    return pd.concat(scores).sort_index().to_frame()
