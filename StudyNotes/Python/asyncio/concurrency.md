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
