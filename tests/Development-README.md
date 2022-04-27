# Steps to set up testing environment

Set up conda environment:
```
conda create --name picai_eval python=3.9
```

Activate environment:
```
conda activate picai_eval
```

Install module and dependencies:
```
pip install -e .
pip install -r requirements_dev.txt
```

Perform tests:
```
pytest
mypy src
flake8 src
```
