# Занятие 14

## Домашнее задание 2 (продолжение)

**!!! вопрос 5**. Посчитайте производную через `Batchnorm1d`.

**Срок сдачи 20 декабря 2021 23:59**

## Домашнее задание 3

Натренировать модель для [задачи классификации
изображений](https://www.kaggle.com/c/recursion-cellular-image-classification/overview)

[Данные](https://www.kaggle.com/c/recursion-cellular-image-classification/data)
(нужно будет зарегистрироваться).

[Примеры
решений](https://www.kaggle.com/c/recursion-cellular-image-classification/code)
(копировать не нужно, но можно адаптировать чужой пайплайн под свой)

[Обсуждения](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion) (полезно знать что пишут другие).

Пакет [`timm`](https://github.com/rwightman/pytorch-image-models) с различными
архитектурами свёрточных сетей

```bash
pip isntall timm
```

Рекомендую воспользоваться следующими архитектурами:

- `resnet`
- `densenet`
- `efficientnet`

В качестве решения ожидается ссылка на вашем `github` (или публичный, или с
доступом для `kbrodt`) с обученной моделью и скриптом (`.py`, `.ipynb`) для
запуска на данных из конкурса. Сделать `Late Submission` и прикрепить скриншот,
где видно, что это вы (нажать, например, на гуся слева сверху).

**Срок сдачи 24 декабря 2021 23:59**
