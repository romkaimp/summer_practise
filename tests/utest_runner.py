import unittest
import utest_SLAE

test_loader = unittest.TestLoader()

suite = test_loader.loadTestsFromTestCase(utest_SLAE.SLAE1Tests)

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
