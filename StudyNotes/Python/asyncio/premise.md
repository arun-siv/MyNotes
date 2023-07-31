### Premise

```python
print("Let's take a look!")
```

```zsh
python -m pip install fastapi uvicorn requests
```

```python
from fastapi import FastAPI
from uvicorn import run

app = FastAPI()

@app.get('/test')
async def test():
    return {'success': True}

if __name__ == '__main__':
    from threading import Thread
    def target():
        from requests import get
        sleep(1)
        print(get('http://localhost:8000/test'))
    thread = Thread(target=target)
    thread.start()
    run(app)
    thread.join()
```

```zsh
http get http://localhost:8000/test
```

```zsh
python -m pip install numpy pandas matplotlib scipy
```

```python
from numpy.random import default_rng
from pandas import DataFrame, MultiIndex, date_range, Series, to_timedelta, IndexSlice, CategoricalIndex
from pathlib import Path
from sys import exit
from scipy.stats import skewnorm
import sys; sys.breakpointhook = exit

data_dir = Path('data')
data_dir.mkdir(exist_ok=True, parents=True)

full_dates = date_range('2000-01-01', periods=365*5, freq='D')
dates = date_range('2000-01-01', periods=365*5, freq='D')

assets = '''
    Equipment Medicine Metals Software StarGems Uranium
'''.split()
assets = CategoricalIndex(assets)

stars = '''
    Sol
    Boyd Fate Gaol Hook Ivan Kirk Kris Quin
    Reef Sand Sink Stan Task York
'''.split()
stars = CategoricalIndex(stars)

players = '''
    Alice Bob Charlie Dana
'''.split()
players = CategoricalIndex(players)

rng = default_rng(0)

inventory = (
    Series(
        index=(idx :=
            MultiIndex.from_product([
                players,
                assets,
            ], names='player asset'.split())
        ),
        data=rng.normal(loc=1, scale=.25, size=len(idx)),
        name='volume',
    ) * Series({
        'Equipment': 1_000,
        'Medicine':    500,
        'Metals':    1_250,
        'Software':    350,
        'StarGems':      5,
        'Uranium':      50,
    }, name='volume').rename_axis(index='asset')
).pipe(lambda s:
    s
        .sample(len(s) - 3, random_state=rng)
        .sort_index()
).pipe(lambda s:
    s
        .astype('int')
        .reindex(idx)
        .astype('Int64')
)

base_prices = Series({
    'Equipment':    7,
    'Medicine':    40,
    'Metals':       3,
    'Software':    20,
    'StarGems': 1_000,
    'Uranium':    500,
}, name='price').rename_axis('asset')

price_shifts = (
    Series(
        index=(idx :=
            MultiIndex.from_product([
                full_dates,
                stars,
                assets,
            ], names='date star asset'.split())
        ),
        data=(
            rng.normal(loc=1, scale=0.05, size=(len(stars), len(assets))).clip(0, 1.5)
            *
            rng.normal(loc=1, scale=0.02, size=(len(full_dates), len(stars), len(assets))).clip(0, 1.5).cumprod(axis=0)
        ).ravel(),
        name='price',
    )
)
spreads = (
    Series(
        index=(idx :=
            MultiIndex.from_product([
                full_dates,
                stars,
                assets,
            ], names='date star asset'.split())
        ),
        data=skewnorm(a=1, loc=.02, scale=.01).rvs(len(idx), random_state=rng).clip(-0.01, +.05),
        name='price',
    )
)

market = DataFrame({
    'buy':  base_prices * price_shifts * (1 + spreads),
    'sell': base_prices * price_shifts,
}).rename_axis(columns='direction').pipe(
    lambda df: df.set_axis(
        df.columns.astype('category'),
        axis='columns',
    )
)

loc_ps = {
    pl: (p := rng.integers(10, size=len(stars))) / p.sum()
    for pl in players
}
locations = (
    DataFrame(
        index=(idx := dates),
        data={
            pl: rng.choice(stars, p=loc_ps[pl], size=len(idx))
            for pl in players
        },
    )
    .rename_axis(index='date', columns='player')
    .pipe(lambda s:
        s
        .set_axis(
            s.columns.astype(players.dtype),
            axis='columns',
        )
        .astype(
              stars.dtype,
        )
    )
    .stack('player')
    .rename('star')
    .pipe(
        lambda s: s
            .sample(frac=.75, random_state=rng)
            .reindex(s.index)
            .groupby('player').ffill()
            .groupby('player').bfill()
            .sort_index()
    )
)

trips = (
    locations.groupby('player', group_keys=False).apply(
        lambda g: g[g != g.shift()]
    ).sort_index()
)

standard_volumes = (10_000 / base_prices).round(0)

trades = (
    DataFrame(
        index=(idx :=
            MultiIndex.from_product([
                dates,
                players,
                assets,
                range(25),
            ], names='date player asset trade#'.split())
        ),
        data={
            'sentiment': rng.normal(loc=0, scale=.025, size=len(idx)),
            'regret': rng.normal(loc=0, scale=.0005, size=len(idx)),
            'edge': rng.normal(loc=1, scale=.001, size=len(idx)).clip(.75, 1.25),
        },
    )
    .pipe(
        lambda df: df
            .assign(
                buy=lambda df: (df.groupby(['player', 'asset'])['sentiment'].rolling(3).mean() > 0).values,
                sign=lambda df: df['buy'] * -1 + ~df['buy'],
                direction=lambda df: df['buy'].map({True: 'buy', False: 'sell'}).astype(market.columns.dtype),
                volume=lambda df: df['sign'] * rng.normal(loc=1, scale=.5, size=len(df)).clip(0, 2),
            )
            .assign(
                star=lambda df:
                    locations.loc[
                        MultiIndex.from_arrays([
                            df.index.get_level_values('date'),
                            df.index.get_level_values('player'),
                        ])
                    ].values,
                asset_price=lambda df: (
                    market.stack('direction').loc[
                        MultiIndex.from_arrays([
                            df.index.get_level_values('date'),
                            df['star'],
                            df.index.get_level_values('asset'),
                            df['direction'],
                        ])
                    ].values
                ),
                price=lambda df: df['asset_price'] * df['edge'],
                mark=lambda df: df['price'] * (1 + df['regret']),
                volume=lambda df: (df['volume'] * standard_volumes).round(-1).astype(int),
            )
    )
    .pipe(
        lambda df: df
            .loc[lambda df: df['volume'] != 0]
            .sample(frac=.5, random_state=rng)
            .sort_index()
    )
)

data_dir = Path('data')
data_dir.mkdir(exist_ok=True, parents=True)

# market.to_csv(data_dir / 'real-market.csv')
# market.to_pickle(data_dir / 'real-market.pkl')
# market.round(2).to_csv(data_dir / 'market.csv')
market.round(2).to_pickle(data_dir / 'market.pkl.zstd')

# trips.to_csv(data_dir / 'trips.csv')
# trips.to_pickle(data_dir / 'trips.pkl')

# inventory.to_csv(data_dir / 'inventory.csv')
# inventory.to_pickle(data_dir / 'inventory.pkl')

# locations.to_csv(data_dir / 'locations.csv')
# locations.to_pickle(data_dir / 'locations.pkl')

# trades.to_csv(data_dir / 'real-trades.csv')
# trades.to_pickle(data_dir / 'real-trades.pkl')
# trades.droplevel('trade#')[['volume', 'price']].to_csv(data_dir / 'trades.csv')
trades.droplevel('trade#')[['volume', 'price']].to_pickle(data_dir / 'trades.pkl.zstd')
# trades.droplevel('trade#')['mark'].round(4).to_csv(data_dir / 'marks.csv')
# trades.droplevel('trade#')['mark'].round(4).to_pickle(data_dir / 'marks.pkl')

print(
    market.sample(3).sort_index(),
    # trips.sample(3).sort_index(),
    # inventory.sample(3).sort_index(),
    # locations.sample(3).sort_index(),
    # trades[['volume', 'price']].sample(3).sort_index(),
    f'{len(trades) = }',
    sep='\n{}\n'.format('\N{box drawings light horizontal}' * 40),
)
```
