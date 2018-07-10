
## Release

    - git flow release start <VERSION>
    - version bump motmetrics/__init__.py
    - conda env create -f environment.yml
    - activate motmetrics-env
    - [pip install lapsolver]
    - pip install .
    - pytest
    - deactivate
    - conda env remove -n motmetrics-env
    - git add, commit
    - git flow release finish <VERSION>
    - git push
    - git push --tags
    - git checkout master
    - git push
    - git checkout develop
    - check appveyor, travis and pypi