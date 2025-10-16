import sys
import unittest

sys.path.append('src/') # noqa

#import random
import utils


# This unit testing WILL be executed by GitHub actions
class TestDataExtractor(unittest.TestCase):
	
    # This is just a dummy test for now, but we'll make real tests
    def test_DummyTest(self):

        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
