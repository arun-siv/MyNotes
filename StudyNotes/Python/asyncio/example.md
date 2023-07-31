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

