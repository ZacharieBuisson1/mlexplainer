"""Test suite that runs all tests in the package."""

import unittest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all test modules
from tests.test_core import TestBaseMLExplainer
from tests.test_utils import TestDataProcessing, TestQuantiles
from tests.test_shap_wrapper import TestShapWrapper
from tests.test_binary_explainer import TestBinaryMLExplainer
from tests.test_multilabel_explainer import TestMultilabelMLExplainer
from tests.test_imports import TestPackageImports

# Import legacy tests for compatibility
from tests.test_shap_explainer import TestShapExplainer, TestNbMinQuantiles, TestGroupValues, TestBinaryMLExplainerImports


def create_test_suite():
    """Create a comprehensive test suite."""
    suite = unittest.TestSuite()
    
    # Core functionality tests
    suite.addTest(unittest.makeSuite(TestBaseMLExplainer))
    
    # Utility tests
    suite.addTest(unittest.makeSuite(TestDataProcessing))
    suite.addTest(unittest.makeSuite(TestQuantiles))
    
    # SHAP wrapper tests
    suite.addTest(unittest.makeSuite(TestShapWrapper))
    
    # Explainer tests
    suite.addTest(unittest.makeSuite(TestBinaryMLExplainer))
    suite.addTest(unittest.makeSuite(TestMultilabelMLExplainer))
    
    # Import tests
    suite.addTest(unittest.makeSuite(TestPackageImports))
    
    # Legacy tests (for backward compatibility)
    suite.addTest(unittest.makeSuite(TestShapExplainer))
    suite.addTest(unittest.makeSuite(TestNbMinQuantiles))
    suite.addTest(unittest.makeSuite(TestGroupValues))
    suite.addTest(unittest.makeSuite(TestBinaryMLExplainerImports))
    
    return suite


def run_all_tests():
    """Run all tests in the suite."""
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)