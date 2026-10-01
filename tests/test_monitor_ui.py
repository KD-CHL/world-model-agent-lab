import http.client
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest

from wmal.monitor.catalog import RunCatalog
from wmal.monitor.server import MonitorServer


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'script' and 'src' in values:
            self.assets.append(values['src'])
        if tag == 'link' and values.get('rel') == 'stylesheet':
            self.assets.append(values['href'])


class MonitorUITests(unittest.TestCase):
    def test_browser_entry_serves_all_local_assets_with_expected_content_types(self):
        with tempfile.TemporaryDirectory() as directory:
            server = MonitorServer(RunCatalog(directory), port=0).start()
            try:
                connection = http.client.HTTPConnection(*server.address)
                connection.request('GET', '/')
                response = connection.getresponse()
                self.assertEqual(response.headers['Content-Type'], 'text/html; charset=utf-8')
                parser = AssetParser()
                parser.feed(response.read().decode('utf-8'))
                self.assertGreaterEqual(len(parser.assets), 2)
                for asset in parser.assets:
                    self.assertTrue(asset.startswith('/static/'))
                    client = http.client.HTTPConnection(*server.address)
                    client.request('GET', asset)
                    loaded = client.getresponse()
                    self.assertEqual(loaded.status, 200)
                    self.assertGreater(len(loaded.read()), 100)
                    client.close()
                connection.close()
            finally:
                server.shutdown()
