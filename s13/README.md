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

## Домашнее задание

1. Для функции `f(x, y) = x^2 / y + exp(y - x)` выписать граф вычислений и все
   производные для каждого узла в графе.

2. Заполнить файл [`tor4.py`](./tor4.py) так, чтобы все тесты позеленели.

Методы `sub`, `mul`, `pow`, `sigmoid` являются покомпонентными, то есть в
случае входных аргументов `x = (x_1, x_2, ..., x_n)` и `y = (y_1, y_2, ...,
y_n)` (могут быть скалярами или матрицами) метод `mul(x, y)` возвращает `x * y
= (x_1 * y_1, x_2 * y_2, ..., x_n * y_n)`. Производные `left` и `right`
означают частные производные от выхода (`x * y`) по левому (`x`) и правому
(`y`) аргументам соответственно.

Для каждого метода нужно выписать производную на бумажке и потом реализовать в
коде.

*Подсказка*:
- Смотрите на тесты
- Чтобы посчитать производную по `x` от матричного умножения `Ax` проще
  смотреть на это выражение покомпонентно и после вычислений собрать обратно в
  матричный вид.

3. Заполнить файл [`training_loop.py`](./training_loop.py) и обучить модель классификации.

В качестве решения ожидается ссылка на вашем `github` (или публичный, или с
доступом для `kbrodt`) с кодом и файлами `pdf` (или фотографиями `jpeg`, `png`,
...) вычислений всех производных.

**Срок сдачи 13 декабря 2021 23:59**
