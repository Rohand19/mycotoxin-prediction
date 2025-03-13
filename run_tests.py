#!/usr/bin/env python3
import os
import sys
import unittest

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Run the tests
if __name__ == "__main__":
    # Load the test module
    test_module = unittest.TestLoader().discover('tests')
    
    # Run the tests
    unittest.TextTestRunner().run(test_module) 