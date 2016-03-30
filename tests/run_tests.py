import glob
import unittest

testmodules = [testfile.replace('.py', '') for testfile in glob.glob('test_*') if not '.pyc' in testfile]

suite = unittest.TestSuite()

for t in testmodules:
    suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

unittest.TextTestRunner().run(suite)
