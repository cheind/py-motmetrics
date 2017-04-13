
## Release

    - git flow release start <VERSION>
    - version bump motmetrics/__init__.py
    - conda env create -f environment.yml
    - activate motmetrics-env
    - pip install .
    - pytest
    - deactivate motmetrics-env
    - conda env remove -n motmetrics-env
    - clean dist/ dir
    - python setup.py bdist_wheel
    - twine upload dist\*
    - git add, commit
    - git flow release finish <VERSION>
    - git push
    - git push --tags
    - git checkout master
    - git push
    - git checkout develop