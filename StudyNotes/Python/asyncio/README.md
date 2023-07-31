# Asynchronous Python with `asyncio`

## A Seminar by ‘Don’t Use This Code’

![Logo: Don’t Use This Code, LLC](logo-small.png)

**Presenter**: James Powell <james@dutc.io>

<div style="display: flex; justify-content: center; font-size: 2em; width: auto; padding: .25em 5em .25em 5em;">
    <p style="text-align: center">
        Join us on <a href="https://discord.gg/ZhJPKYSfNp">Discord (https://discord.gg/ZhJPKYSfNp)</a> for discussion and guidance!
    </p>
</div>

## Contents

* [A Seminar by ‘Don’t Use This Code’](#a-seminar-by-‘don’t-use-this-code’)
* [Book a Class!](#book-a-class!)
* [Notes](#notes)
* [About](#about)
  * [Don’t Use This Code; Training & Consulting](#don’t-use-this-code;-training-&-consulting)

## Book a Class!

<big><big>Book a class or training for your team!</big></big>

Please reach out to us at [learning@dutc.io](mailto:learning@dutc.io) if are
interested in bringing this material, or any of our other material, to your
team.

We have courses on topics such as:
- intro Python
- expert Python
- data engineering with Python
- data science and scientific computing with `numpy`, `pandas`, and `xarray`

If you reach out to us, we can also provide a printable copy of the notes,
cleaned-up and in .pdf format, as well as a professionally edited video
recording of this presentation.

## Notes

<!--
<div style="display: flex; justify-content: center; font-size: 2em; width: auto; padding: .25em 5em .25em 5em;">
    <p style="text-align: center">
        <a href="materials/2023-07-10-asyncio-data.zip">Data Files (2023-07-10-asyncio-data.zip)</a>
    </p>
</div>
-->

### Context

> Star Trader is a 1974 video game and an early example of the space trading
> genre. The game involves players moving from star to star on a map of the
> galaxy, buying and selling quantities of six types of merchandise in a
> competition to make the most money. The game was developed by Dave Kaufman
> for computers in 1973, and its BASIC source code was printed in the January
> 1974 issue of the People’s Computer Company Newsletter. It was reprinted in
> the 1977 book What to Do After You Hit Return. The game was the inspiration
> for the multiplayer Trade Wars series, beginning in 1984, and is thought to
> be the antecedent to much of the space trading genre.

— [*Star Trader* on Wikipedia.org](https://en.wikipedia.org/wiki/Star_Trader)

![Star Trader, 1974](https://upload.wikimedia.org/wikipedia/en/d/d2/Star_Trader_1974_screenshot.png)
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
### Generators

```python
print("Let's take a look!")
```

```python
def f(a, b, c):
    return a + b + c
```

```python
def f(data, *, mode):
    if mode:
        return data ** 2
    return data ** 3

print(f'{f(123, mode=True)  = :,}')
print(f'{f(123, mode=False) = :,}')
```

```python
from pandas import read_csv

help(read_csv)

read_csv('data.csv', header=None, skipfooter=10, parse_dates=['date'])
```

```python
def f(): # function
    pass

def g(): # function ⇒ generator
    return
    yield

print(f'{f   = }')
print(f'{f() = }') # f() ⇒ result

print(f'{g   = }')
print(f'{g() = }') # g() ⇒ intermediation for getting results
```

Terminology (human)

```python
# function
# “subroutine”
# ⇒ result
def f(): pass

# “generator”
# ⇒ “generator instance”
# ⇒ next(...) ⇒ result
def g(): yield
```

```python
def f(data, *, mode):
    rv = []
    if mode:
        rv.append(data ** 2)
    rv.append(data ** 3)
    return rv

print(f'{f(123, mode=True) = }')

# for rv in f(123, mode=True):
#     print(f'{rv = :,}')

def g(data, *, mode):
    if mode:
        yield data ** 2
    yield data ** 3

gi = g(123, mode=True)
print(f'{next(gi) = :,}')
...
print(f'{next(gi) = :,}')

# for rv in g(123, mode=True):
#     print(f'{rv = :,}')

# for x in xs:
#    pass
# xi = iter(xs)
# while True:
#     try:
#         x = next(xi)
#     except StopIteration:
#         break
```

```python
from random import Random
from collections import namedtuple, defaultdict
from string import ascii_lowercase
from statistics import mean, pstdev

rnd = Random(0)

class Entity(namedtuple('EntityBase', 'name value')):
    @classmethod
    def from_random(cls, *, random_state=None):
        random_state = random_state if random_state is not None else Random()
        return cls(
            name=''.join(rnd.choices(ascii_lowercase, k=4)),
            value=rnd.randint(-1_000, +1_000),
        )

class Analysis:
    def __init__(self, random_state=None):
        self.random_state = random_state if random_state is not None else Random()

    def clean(self, num_std=1):
        m, sd = mean(ent.value for ent in self.data), pstdev(ent.value for ent in self.data)
        self.clean_data = [
            ent for ent in self.data
            if m - sd * num_std <= ent.value <= m + sd * num_std
        ]
        return self.clean_data
    def load(self):
        self.data = [
            Entity.from_random(random_state=self.random_state)
            for _ in range(100)
        ]
        return self.data
    def summarize(self):
        by_name = defaultdict(list)
        for ent in self.results:
            by_name[ent.name].append(ent.value)
        return {k: mean(v) for k, v in by_name.items()}
    def compute(self):
        self.results = [Entity(ent.name, ent.value**2) for ent in self.clean_data]
        return self.results

obj = Analysis()
data = obj.load()
clean_data = obj.clean(num_std=3)
results = obj.compute()
summary = obj.summarize()
print(f'{summary = }')
```

```python
from random import Random
from collections import namedtuple, defaultdict
from string import ascii_lowercase
from statistics import mean, pstdev

class Entity(namedtuple('EntityBase', 'name value')):
    @classmethod
    def from_random(cls, *, random_state=None):
        random_state = random_state if random_state is not None else Random()
        return cls(
            name=''.join(rnd.choices(ascii_lowercase, k=4)),
            value=rnd.randint(-1_000, +1_000),
        )

def analysis(random_state=None):
    random_state = random_state if random_state is not None else Random()

    data = [
        Entity.from_random(random_state=random_state)
        for _ in range(100)
    ]
    yield data

    m, sd = mean(ent.value for ent in data), pstdev(ent.value for ent in data)
    clean_data = [
        ent for ent in data
        if m - sd <= ent.value <= m + sd
    ]
    yield clean_data

    results = [Entity(ent.name, ent.value**2) for ent in clean_data]
    yield results

    by_name = defaultdict(list)
    for ent in results:
        by_name[ent.name].append(ent.value)
    yield {k: mean(v) for k, v in by_name.items()}

rnd = Random(0)
obj = analysis(random_state=rnd)
print(f'{next(obj) = }')
print(f'{next(obj) = }')
print(f'{next(obj) = }')
print(f'{next(obj) = }')
```

Terminology: “generator coroutine”

```python
def g(data, *, mode):
    for x in data:
        yield x ** 2 if mode else x ** 3

gi = g(range(100_000_000_000_000_000_000_000_000_000), mode=True)
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
gi.close()
```

```python
def g():
    yield 1
    yield 2
    yield 3

gi = g()
# gi.throw(ValueError())
print(f'{gi.send(None) = }') # next(gi) ⇒ gi.send(None)
print(f'{gi.send(None) = }')
print(f'{gi.send(None) = }')
# gi.close()

# print(
#     dir(g())
# )
```

```python
def f():
    while True:
        ...

f()
...
```

```python
# “generator coroutine”
def count():
    x = 0
    while True:
        yield x
        x += 1

gi = count()
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
```

```python
from itertools import count

gi = count()
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
```

```python
# “subroutine”
def f(data):
    rv = ...
    return rv

# “generator coroutine”
def count(start=0):
    x = start
    while True:
        if (yield x):
            x = start
        else:
            x += 1

gi = count(0)
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{gi.send(True) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
print(f'{next(gi) = }')
```
### Concurrency

```python
print("Let's take a look!")
```

```zsh
typeset -a computations=(
    '1+1'
    '2*2'
    '3**3'
)
for c in "${(@)computations}"; do
    time python -c "__import__('time').sleep(1); print(${c})"
done

wait
```

```python
from multiprocessing import Process, Queue
from itertools import product
from time import sleep

def target(x, y):
    sleep(1)
    return x ** y

pool = [
    Process(target=target, kwargs={'x': x, 'y': y})
    for x, y in
    product(range(3), repeat=2)
]
for x in pool: print(f'{x.start() = }')
for x in pool: print(f'{x.join() = }')
```

```python
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from collections import namedtuple
from time import sleep, perf_counter
from contextlib import contextmanager

@contextmanager
def timed(msg):
    before = perf_counter()
    yield
    after = perf_counter()
    print(f'{msg:<48} \N{mathematical bold capital delta}t: {after - before:.4f}s')

def target(d):
    sleep(1)
    return d.x ** d.y

Data = namedtuple('Data', 'x y')
dataset = [Data(x, y) for x, y in product(range(3), repeat=2)]
with timed('ProcessPoolExecutor(max_workers=5)'):
    with ProcessPoolExecutor(max_workers=len(dataset)) as pool:
        results = pool.map(target, dataset)
        # print(f'{dataset = }')
        print(f'{dict(zip(dataset, results)) = }')
```

```python
from threading import Thread
from time import sleep
from itertools import product

def target(x, y):
    sleep(1)
    print(f'{x ** y = }')

pool = [
    Thread(target=target, kwargs={'x': x, 'y': y})
    for x, y in product(range(3), repeat=2)
]
for x in pool: x.start()
for x in pool: x.join()
```

```python
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from collections import namedtuple
from time import sleep, perf_counter
from contextlib import contextmanager

@contextmanager
def timed(msg):
    before = perf_counter()
    yield
    after = perf_counter()
    print(f'{msg:<48} \N{mathematical bold capital delta}t: {after - before:.4f}s')

def target(d):
    sleep(1)
    print(f'{d.x ** d.y = }')

Data = namedtuple('Data', 'x y')
dataset = [Data(x, y) for x, y in product(range(3), repeat=2)]
with timed('ThreadPoolExecutor(max_workers=len(dataset))'):
    with ThreadPoolExecutor(max_workers=len(dataset)) as pool:
        results = pool.map(target, dataset)
        # print(f'{dataset = }')
        print(f'{dict(zip(dataset, results)) = }')
```

```python
from threading import Thread
from multiprocessing import Process
from time import sleep

# xs = [*range(10)]
d = {}

def target1():
    while True:
        # import pandas
        print(f'thread1 {len(d) = }')
        d[len(d)] = None
        sleep(1)

def target2():
    while True:
        # import pandas
        print(f'thread2 {len(d) = }')
        sleep(1)

pool = [
    Thread(target=target1),
    Thread(target=target2),
]
for x in pool: x.start()
for x in pool: x.join()
```

```zsh
time python -c 'import pandas'
```

```python
def _():
    d[len(d)] = None

from dis import dis
dis(_)

d[len(d)] = None
# LOAD_GLOBAL
# ...
# STORE_//SUBSCR
# # PyObject_//SetItem
# # # PyDict_//SetItem
# # # # insert//dict
# # # # # mov
# # # # # xor
# # # # # ~~~~~~
# # # # # load
# # # # # store
# ...
```

“Global Interpreter Lock” (“GIL”)

```python
from threading import Thread, Lock
from time import sleep
from random import Random

rnd = Random(0)
lock = Lock()

def f(k):
    return k + 1

def target1():
    owned = _owned[target1]
    while True:
        print(f'thread1 {shared = } {owned = }')
        with lock:
            k = rnd.choice([*shared.keys()])
            v = shared[k]
            shared[k] = owned[k] = f(v)
        sleep(1)

def target2():
    owned = _owned[target2]
    while True:
        print(f'thread2 {shared = } {owned = }')
        with lock:
            k = rnd.choice([*shared.keys()])
            v = shared[k]
            shared[k] = owned[k] = f(v)
        sleep(1)

shared = {'a': 1, 'b': 2, 'c': 3}
_owned = {
    target1: {'a': 1, 'b': 2, 'c': 3},
    target2: {'a': 1, 'b': 2, 'c': 3},
}

pool = [
    Thread(target=target1),
    Thread(target=target2),
]
for x in pool: x.start()
for x in pool: x.join()
```

Threading
- “preëmptively schedule” & data is shared by default

Multiprocessing
- “preëmptively schedule” & data is isolated by default

```python
from multiprocessing import Process, Queue
from time import sleep
from random import Random

rnd = Random(0)

def producer(q):
    while True:
        q.put(rnd.random())
        sleep(1)

def consumer(q):
    while True:
        print(f'{q.get() = }')
        sleep(1)

q = Queue()
pool = [
    Process(target=producer, kwargs={'q': q}),
    Process(target=consumer, kwargs={'q': q}),
]
for x in pool: x.start()
for x in pool: x.join()
```

```python
from pickle import dumps
from sqlite3 import connect

# def f():
#     def g():
#         pass
#     return g

def g():
    yield

# with open(__file__) as f:
with connect(':memory:') as conn:
    print(
        f'{dumps(123)   = }',
        f'{dumps("abc") = }',
        # f'{dumps(f())   = }',
        # f'{dumps(g()) = }',
        # f'{dumps(f) = }',
        f'{dumps(conn) = }',
        sep='\n',
    )
```

```python
from queue import Queue
```

```python
from time import sleep

def task(name):
    while True:
        print(f'task {name = }')
        yield
        sleep(1)

ti1 = task('aaa')
ti2 = task('bbb')
next(ti1)
next(ti2)
next(ti1)
next(ti2)
```

```python
from time import sleep, perf_counter
from random import Random
from collections import deque

def scheduler(*tasks):
    times = {t: 0 for t in tasks}
    while True:
        ready = {*()}
        for t in tasks:
            if times[t] <= max(times.values()):
                ready.add(t)

        for t in ready:
            before = perf_counter()
            next(t)
            after = perf_counter()
            times[t] += after - before

d = {'a': 1, 'b': 2, 'c': 3}

def task(name):
    while True:
        print(f'task {name = }')
        d[k] = d[k := rnd.choice([*d.keys()])] + 1
        yield
        sleep(rnd.random())

rnd = Random()
scheduler(
    task('aaa'),
    task('bbb'),
    task('ccc'),
)
```

```python
from time import sleep, perf_counter
from random import Random
from collections import deque
from asyncio import run, gather, sleep as aio_sleep

d = {'a': 1, 'b': 2, 'c': 3}

async def task(name):
    while True:
        print(f'task {name = }')
        d[k] = d[k := rnd.choice([*d.keys()])] + 1
        await aio_sleep(rnd.random())

async def main():
    await gather(
        task('aaa'),
        task('bbb'),
        task('ccc'),
    )


rnd = Random()
run(main())
```

```python
async def f():
    async with ...:
        ...
    async for ...:
        ...

from asyncio import create_task, TaskGroup
```
### `async` & `await`

```python
print("Let's take a look!")
```

```python
def f(): return # “subroutine”

def g(): yield  # “generator”

def g(): # “(generator) coroutine”
    _ = yield

async def t(): # “asynchronous {task,function}”
    pass
```

```python
async def task(name):
    while True:
        print('task {name = }')

task('aaa')
```

```python
from asyncio import run
from time import sleep

async def task(name):
    while True:
        print(f'task {name = }')
        sleep(1)

run(task('aaa'))
```

```python
from asyncio import run, gather
from time import sleep

async def task(name):
    while True:
        print(f'task {name = }')
        sleep(1)

async def f(x):
    return x ** 2

async def main():
    results = await gather(*(f(x) for x in range(3)))
    print(f'{results = }')

run(main())
```

```python
from asyncio import run, gather, sleep as aio_sleep
from time import sleep

async def task(name):
    while True:
        print(f'task {name = }')
        await aio_sleep(0)

async def main():
    await gather(
        task('aaa'),
        task('bbb'),
        task('ccc'),
    )

run(main())
```

“Function coloring”

```python
from asyncio import get_event_loop
# “scheduler ignorant”
def f1():
    f2()        # ok!      “scheduler ignorant”
    await af2() # NOT OK!! “scheduler aware”
    # loop = get_event_loop().create_task(af2())
    print(f'{loop = }')

# “scheduler aware”
async def af1():
    f1()        # ok! “scheduler ignorant”
    await af2() # ok! “scheduler aware”

async def af2():
    pass
def f2():
    pass

f1()
```

```python
class ctx:
    def __enter__(self):
        print('before')
    def __exit__(self, *_):
        print('after')

with ctx():
    print('inside')
```

```python
from contextlib import contextmanager

@contextmanager
def ctx():
    print('before')
    yield
    print('after')

with ctx():
    print('inside')
```

```python
from asyncio import run
from contextlib import contextmanager

class ctx:
    def __enter__(self):
        return ...
    def __exit__(self, *_): pass

@contextmanager
def ctx():
    yield ...

@lambda main: run(main())
async def main():
    print(f'main')

    with ctx() as ctxobj:
        print(f'{ctxobj = }')
```

Threading:
- everything operates within the same process
- we lock per bytecode
- NO speedup for number-crunching-style operations
- waiting for external
- I/O-bound

Multiprocessing:
- we have multiple, isolated processes
- no GIL locking
- number crunching
- compute-bound

```python
def target():
    (x + y * z) // w
```

Time is Spent:
- performing computations ⇒ compute-bound
- waiting for external resources ⇒ I/O-bound

```python
from requests import get

get(...) # 2s
get(...)
get(...)
```

```zsh
python -m pip install httpx
```

```python
from fastapi import FastAPI
from uvicorn import run
from time import sleep
from multiprocessing import Process
from httpx import AsyncClient
from asyncio import run as aio_run, gather, sleep as aio_sleep

def producer():
    app = FastAPI()

    @app.get('/data')
    async def data():
        await aio_sleep(1)
        return {'success': True}

    run(app)

def consumer():
    sleep(.1)
    @lambda main: aio_run(main())
    async def main():
        async with AsyncClient() as client:
            results = await gather(
                client.get('http://localhost:8000/data'),
                client.get('http://localhost:8000/data'),
                client.get('http://localhost:8000/data'),
            )
            print(f'{results = }')

pool = [
    Process(target=producer),
    Process(target=consumer),
]

for x in pool: x.start()
for x in pool: x.join()
```

```python
from httpx import AsyncClient

class _AsyncClient:
    async def __enter__(self):
        ...
    async def __exit__(self, *_):
        ...

async def f():
    with AsyncClient() as client:
        await client.get(...)
        await client.post(...)
```

```python
with ctx() as ctxobj:
    pass

ctxmgr = ctx()
ctxmgr.__enter__()
try:
    ...
except Exception as e:
    ctxmgr.__exit__(e, ..., ...)
else:
    ctxmgr.__exit__(None, None, None)
```

```python
await with ctx() as ctxobj:
    pass

ctxmgr = ctx()
await ctxmgr.__enter__()
try:
    ...
except Exception as e:
    await ctxmgr.__exit__(e, ..., ...)
else:
    await ctxmgr.__exit__(None, None, None)
```

```python
from asyncio import run

async def task(msg):
    print(f'{msg}')

class actx:
    async def __aenter__(self):
        await task(msg='before')
    async def __aexit__(self, *_):
        await task(msg='after')

@lambda main: run(main())
async def main():
    async with actx():
        print('inside')
```

```zsh
python -m pip install aiosqlite aiofiles
```

```python
from aiosqlite import connect
from asyncio import run, create_task, sleep as aio_sleep, TaskGroup

async def producer(conn):
    while True:
        print(f'producer {conn = }')
        await conn.execute('insert into test values ("abc", 123)')
        await aio_sleep(1)

async def consumer(conn):
    while True:
        print(f'consumer {conn = }')
        cur = await conn.execute('select sum(value) from test')
        async for row in cur:
            print(f'{row = }')
        await aio_sleep(1)

@lambda main: run(main())
async def main():
    async with connect(':memory:') as conn:
        await conn.execute('''
            create table test (
                name text
              , value number
            )
        ''')
        async with TaskGroup() as tg:
            tg.create_task(producer(conn))
            tg.create_task(consumer(conn))
```

```python
class Cursor:
    def __init__(self):
        self.size = 3
    def __iter__(self):
        return self
    def __next__(self):
        if not self.size:
            raise StopIteration()
        self.size -= 1
        return ...

cur = Cursor()
for row in cur:
    print(f'{row = }')

cur = Cursor()
cur_iter = iter(cur)
while True:
    try:
        row = next(cur_iter)
        print(f'{row = }')
    except StopIteration:
        break
```

```python
from asyncio import run

class Cursor:
    def __init__(self):
        self.size = 3
    def __aiter__(self):
        return self
    async def __anext__(self):
        if not self.size:
            raise StopAsyncIteration()
        self.size -= 1
        return ...

@lambda main: run(main())
async def main():
    cur = Cursor()
    async for row in cur:
        print(f'{row = }')

    cur = Cursor()
    cur_iter = aiter(cur)
    while True:
        try:
            row = await anext(cur_iter)
            print(f'{row = }')
        except StopAsyncIteration:
            break
```

```python
from asyncio import run, TaskGroup, sleep as aio_sleep
from random import Random
from collections import deque
from tempfile import TemporaryFile
from string import ascii_lowercase
from aiofiles import open as aio_open
from aiofiles.tempfile import TemporaryFile as aio_TemporaryFile

rnd = Random(0)

async def g():
    while True:
        yield ''.join(rnd.choices([*({*ascii_lowercase} - {*'aeiou'})], k=4)), rnd.random()

async def producer(data):
    async for x in g():
        data.append(x)
        await aio_sleep(1)

async def consumer(data):
    async with aio_TemporaryFile('wt') as f:
        while True:
            if data:
                x = data.popleft()
                await f.write(f'{x}\n')
                print(f'Wrote {x = }')
            await aio_sleep(0)

@lambda main: run(main())
async def main():
    data = deque()
    async with TaskGroup() as tg:
        tg.create_task(producer(data))
        tg.create_task(consumer(data))

```

```python
def f(): return
def g(): yield
def g(): _ = yield

async def f(): return
async def f(): yield
async def f(): _ = yield
```
### Example

```python
print("Let's take a look!")
```

```zsh
python -m pip install fastapi pandas httpx
python -m pip install sse-starlette httpx-sse
```

```python
from multiprocessing import Process

def producer():
    from fastapi import FastAPI
    from uvicorn import run
    from pandas import Series, MultiIndex
    from contextlib import asynccontextmanager
    from numpy.random import default_rng
    from asyncio import TaskGroup, sleep as aio_sleep
    from pydantic import BaseModel
    from fastapi import BackgroundTasks, Request
    from sse_starlette.sse import EventSourceResponse

    rng = default_rng(0)

    assets = '''
        Equipment Medicine Metals Software StarGems Uranium
    '''.split()

    stars = '''
        Sol
        Boyd Fate Gaol Hook Ivan Kirk Kris Quin
        Reef Sand Sink Stan Task York
    '''.split()

    location = 'Sol'
    inventory = Series({x: 0 for x in assets})
    cash = 1_000_000
    market = Series(
        index=(idx := MultiIndex.from_product([
            stars,
            assets,
        ], names=['star', 'asset'])),
        data=(
            rng.normal(loc=100, scale=50, size=len(assets)).clip(0, 1_000)
            * rng.normal(loc=1, scale=.01, size=len(idx)).clip(.9, 1.2).reshape(-1, len(assets))
        ).ravel(),
    ).round(2)

    async def tick_market():
        # global market
        nonlocal market # because we're inside a function body
        while True:
            market *= rng.normal(loc=1.00, scale=.025, size=len(market)).clip(0.8, 1.2)
            await aio_sleep(3)

    @asynccontextmanager
    async def lifespan(_):
        async with TaskGroup() as tg:
            t = tg.create_task(tick_market())
            yield
            t.cancel()

    api = FastAPI(lifespan=lifespan)

    @api.post('/move')
    async def move():
        pass

    @api.get('/prices')
    async def price_stream(request : Request):
        # global location
        nonlocal location
        async def g(location):
            while True:
                if await request.is_disconnected():
                    break
                yield market.loc[location].to_json()
                await aio_sleep(1)
        return EventSourceResponse(g(location))

    @api.get('/market')
    async def market_():
        return {
            'market': market.unstack('asset').to_json(),
        }

    class Trade(BaseModel):
        asset : str
        amount : int

    async def perform_trade(asset, amount):
        desired_price = market.loc[location, asset]
        await aio_sleep(3)
        # global cash
        nonlocal cash # because we're inside a function body
        actual_price = market.loc[location, asset]
        cash -= actual_price * amount
        inventory.loc[asset] += amount
        slippage = actual_price - desired_price
        print(f'{slippage  = }')

    @api.post('/trade')
    async def trade(trade : Trade, bg_tasks : BackgroundTasks):
        asset, amount = trade.asset, trade.amount
        bg_tasks.add_task(perform_trade, asset=asset, amount=amount)

    @api.get('/status')
    async def status():
        return {
            'location': location,
            'inventory': inventory.to_json(),
        }

    @api.get('/test')
    async def test():
        return {'success': True}

    run(api)

def consumer():
    from httpx import AsyncClient, Client
    from httpx_sse import aconnect_sse
    from asyncio import run, sleep as aio_sleep
    from time import sleep
    from pandas import read_json, DataFrame, Series

    sleep(1)
    @lambda main: run(main())
    async def main():
        root_url = 'http://localhost:8000'
        async with AsyncClient() as client:
            # resp = await client.get(f'{root_url}/test')
            # resp = (await client.get(f'{root_url}/status')).json()
            # location, inventory = resp['location'], read_json(resp['inventory'], typ='series')
            # resp = await client.post(f'{root_url}/trade', json={'asset': 'StarGems', 'amount': 1})
            # while True:
            #     market = read_json(
            #         (await client.get(f'{root_url}/market')).json()['market']
            #     )
            #     resp = (await client.get(f'{root_url}/status')).json()
            #     inventory = read_json(resp['inventory'], typ='series')
            #     print(market, inventory, sep='\n')
            #     await aio_sleep(1)
            best_price = None
            async with aconnect_sse(client, 'get', f'{root_url}/prices') as src:
                async for ev in src.aiter_sse():
                    market = Series(ev.json())
                    print(market)
                    if best_price is None or best_price > market.loc['StarGems']:
                        best_price = market.loc['StarGems']
                        resp = await client.post(f'{root_url}/trade', json={'asset': 'StarGems', 'amount': 1})
                    await aio_sleep(2)

pool = [
    Process(target=producer),
    Process(target=consumer),
]
for x in pool: x.start()
for x in pool: x.join()
```


## About

### Don’t Use This Code; Training & Consulting

Don’t Use This Code is a professional training, coaching, and consulting
company. We are deeply invested in the open source scientific computing
community, and are dedicated to bringing better processes, better tools, and
better understanding to the world.

**Don’t Use This Code is growing! We are currently seeking new partners, new
clients, and new engagements for our expert consulting and training
services.**

Our ideal client is an organization, large or small, using open source
technologies, centering around the PyData stack for scientififc and numeric
computing. Organizations looking to better employ these tools would benefit
from our wide range of training courses on offer, ranging from an intensive
introduction to Python fundamentals to advanced applications of Python for
building large-scale, production systems. Working with your team, we can craft
targeted curricula to meet your training goals. We are also available for
consulting services such as building scientific computing and numerical
analysis systems using technologies like Python and React.

We pride ourselves on delivering top-notch training. We are committed to
providing quality training that is uniquely valuable to each individual
attendee, and we do so by investing in three key areas: our
content, our processes, and our contributors.
