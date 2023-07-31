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
