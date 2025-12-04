import sys
import os
import io
import unittest
from unittest import mock

sys.path.append('src/')  # noqa

import visualization.vis as vis


class TestHashTable(unittest.TestCase):

    def test_insert_and_get(self):
        ht = vis.HashTable(capacity=11)
        ht.insert('a', 1)
        ht.insert('b', 2)
        self.assertEqual(ht.get('a'), 1)
        self.assertEqual(ht.get('b'), 2)
        self.assertIsNone(ht.get('nope'))

    def test_overwrite(self):
        ht = vis.HashTable(capacity=7)
        ht.insert('k', 10)
        ht.insert('k', 20)
        self.assertEqual(ht.get('k'), 20)


class TestBenchmarkFunction(unittest.TestCase):

    def test_benchmark_runs_and_outputs(self):
        # Capture stdout
        buf = io.StringIO()
        with mock.patch('sys.stdout', new=buf):
            vis.benchmark_hash_structures(n=200, seed=123)
        out = buf.getvalue()
        self.assertIn('Hash table benchmark (n=200)', out)
        self.assertIn('Custom HashTable:', out)
        self.assertIn('Python dict', out)


if __name__ == '__main__':
    unittest.main()
