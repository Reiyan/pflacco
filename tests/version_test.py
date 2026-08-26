import os
import re

import pflacco

# The version is written twice, in the package and in the packaging metadata.
# This keeps the two from drifting apart on a release.
def test_version_matches_the_packaging_metadata():
    for name, pattern in [('pyproject.toml', r'^version = "([^"]+)"'),
                          ('setup.py', r"version='([^']+)'")]:
        if os.path.exists(name):
            declared = re.search(pattern, open(name).read(), re.M)
            assert declared is not None, f'no version found in {name}'
            assert declared.group(1) == pflacco.__version__
            break
    else:
        raise AssertionError('neither pyproject.toml nor setup.py found')

def test_version_matches_the_documentation_and_citation():
    docs = re.search(r"^release = ['\"]([^'\"]+)['\"]", open(os.path.join('docs', 'source', 'conf.py')).read(), re.M)
    cff = re.search(r'^version: (\S+)', open('CITATION.cff').read(), re.M)
    assert docs.group(1) == pflacco.__version__
    assert cff.group(1) == pflacco.__version__
