# test_imports.py
try:
    import flask
    print("✅ flask imported successfully")
except ImportError as e:
    print(f"❌ flask import failed: {e}")

try:
    import pandas
    print("✅ pandas imported successfully")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import numpy
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import joblib
    print("✅ joblib imported successfully")
except ImportError as e:
    print(f"❌ joblib import failed: {e}")

try:
    import sklearn
    print("✅ scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")

print("\n✅ Import test complete!")
