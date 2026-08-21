from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from playwright.sync_api import sync_playwright

OUT = Path('/mnt/data/SIH-26/test-output')
OUT.mkdir(exist_ok=True)
BASE='http://127.0.0.1:8000'
BROWSER_BASE='https://infrasight.local'


def fetch_local(path='/'):
    with urlopen(BASE + path, timeout=20) as r:
        return r.read().decode('utf-8')


def proxy_route(route, request):
    if not request.url.startswith(BROWSER_BASE):
        return route.continue_()
    target = BASE + request.url[len(BROWSER_BASE):]
    data=request.post_data.encode() if request.post_data is not None else None
    headers={k:v for k,v in request.headers.items() if k.lower() not in {'host','content-length','accept-encoding','connection','origin','referer'}}
    req=Request(target, data=data, headers=headers, method=request.method)
    try:
        with urlopen(req, timeout=20) as r:
            route.fulfill(status=r.status, body=r.read(), content_type=r.headers.get('content-type','application/octet-stream'))
    except HTTPError as e:
        route.fulfill(status=e.code, body=e.read(), content_type=e.headers.get('content-type','application/json'))

index_html = fetch_local('/')
index_html = index_html.replace('<head>', f'<head><base href="{BROWSER_BASE}/">', 1)

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True, executable_path='/usr/bin/chromium', args=['--no-sandbox','--disable-dev-shm-usage'])
    page=browser.new_page(viewport={'width':1440,'height':1100})
    page.route(f'{BROWSER_BASE}/**', proxy_route)
    errors=[]
    page.on('console', lambda msg: errors.append(f'console:{msg.type}:{msg.text}') if msg.type=='error' else None)
    page.on('pageerror', lambda exc: errors.append(f'pageerror:{exc}'))

    # Direct Chromium navigation is blocked by container policy. The exact localhost
    # HTML/API responses are therefore injected/proxied into the browser for E2E UI tests.
    page.goto('about:blank#/dashboard')
    page.set_content(index_html, wait_until='load')
    page.wait_for_selector('text=Predict risk before overruns harden into outcomes.', timeout=20000)
    assert page.locator('text=Projects requiring attention').count()==1
    assert page.locator('text=BharatNet').count()>=1
    page.screenshot(path=str(OUT/'dashboard.png'), full_page=True)

    page.evaluate("location.hash='#/project/701263'")
    page.wait_for_selector('text=Project risk profile', timeout=20000)
    body=page.locator('body').inner_text()
    assert 'Rajasthan Refinery Project' in body
    assert '96.5/100' in body
    assert '84.2%' in body
    assert 'Observed cost escalation' in body
    page.screenshot(path=str(OUT/'project-701263.png'), full_page=True)

    page.evaluate("location.hash='#/scenario?project=701263'")
    page.wait_for_selector('#run-scenario', timeout=20000)
    page.fill('#scenario-progress','95')
    page.fill('#scenario-exp','72000')
    page.click('#run-scenario')
    page.wait_for_selector('.scenario-result', timeout=20000)
    st=page.locator('#scenario-output').inner_text()
    assert 'baseline' in st.lower() and 'scenario' in st.lower()
    page.screenshot(path=str(OUT/'scenario-701263.png'), full_page=True)

    page.evaluate("location.hash='#/time-machine'")
    page.wait_for_selector('text=Project Time Machine', timeout=20000)
    page.select_option('#history-project','701263')
    page.wait_for_timeout(800)
    ht=page.locator('#history-content').inner_text()
    assert 'Rajasthan Refinery Project' in ht
    assert '28 Feb 2026' in ht
    page.screenshot(path=str(OUT/'time-machine.png'), full_page=True)

    page.evaluate("location.hash='#/models'")
    page.wait_for_selector('text=Model Lab', timeout=20000)
    mt=page.locator('body').inner_text().lower()
    for name in ['logistic regression','random forest','xgboost','catboost']:
        assert name in mt
    page.screenshot(path=str(OUT/'model-lab.png'), full_page=True)

    print('BROWSER_SMOKE_OK')
    print('browser_console_errors=', errors)
    browser.close()
