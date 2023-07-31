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
