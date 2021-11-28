# Занятие 13

```bash
virtualenv venv
. ./venv/bin/activate
pip install -r requirements.txt
```

## Автоматическое дифференцирование

Заполнить файл [`tor4.py`](./tor4.py) так, чтобы все тесты стали зелёными

```bash
python -m pytest -v .
# or pytest -v .
```

## Обучение

Заполнить файл [`training_loop.py`](./training_loop.py) и обучить модель классификации.

```bash
python training_loop.py
```
