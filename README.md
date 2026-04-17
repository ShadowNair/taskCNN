# Лабораторная работа №1. Сверточные нейронные сети (CNN)

В проекте реализованы три архитектуры на PyTorch без зависимости от `torchvision`:

- **LeNet-5** для набора **MNIST**
- **VGG16-подобная сеть** для **CIFAR-10**
- **ResNet-34** для **CIFAR-10**

Код соответствует заданию лабораторной работы: можно обучать каждую сеть, сравнивать оптимизаторы, строить графики `Loss` и `Accuracy` по эпохам и отдельно смотреть влияние регуляризации.

## Структура проекта

- `datasets.py` — загрузка и подготовка MNIST и CIFAR-10
- `models.py` — архитектуры LeNet, VGG16 и ResNet-34
- `train.py` — одиночный запуск обучения
- `experiment.py` — серия экспериментов для сравнения оптимизаторов и регуляризации
- `plotting.py` — сохранение графиков и CSV-файлов с историей обучения
- `utils.py` — служебные функции

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Одиночный запуск

### LeNet + Adam
```bash
python train.py --model lenet --optimizer adam --epochs 10 --output-dir ./runs/lenet_adam
```

### VGG16 + NAG
```bash
python train.py --model vgg16 --optimizer nag --epochs 20 --dropout 0.5 --output-dir ./runs/vgg16_nag
```

### ResNet34 + SGD
```bash
python train.py --model resnet34 --optimizer sgd --epochs 20 --dropout 0.3 --output-dir ./runs/resnet34_sgd
```

## Серия экспериментов по оптимизаторам

### LeNet / MNIST
```bash
python experiment.py --model lenet --mode optimizers --epochs 10
```

### VGG16 / CIFAR-10
```bash
python experiment.py --model vgg16 --mode optimizers --epochs 20
```

### ResNet34 / CIFAR-10
```bash
python experiment.py --model resnet34 --mode optimizers --epochs 20
```

После выполнения создаются:

- `summary.csv` — итоговая сводка
- `val_loss_comparison.png` — сравнение `Loss`
- `val_accuracy_comparison.png` — сравнение `Accuracy`
- отдельные папки для каждого оптимизатора с `history.csv`, `metrics.json`, `loss.png`, `accuracy.png`

## Серия экспериментов по регуляризации

Для VGG16 и ResNet-34 удобно сравнивать `dropout`, для LeNet — `weight_decay`.

### LeNet: сравнение weight decay
```bash
python experiment.py --model lenet --mode regularization --optimizer adam --regularizer weight_decay --values 0.0,0.0001,0.0005
```

### VGG16: сравнение dropout
```bash
python experiment.py --model vgg16 --mode regularization --optimizer adam --regularizer dropout --values 0.0,0.3,0.5
```

### ResNet34: сравнение dropout
```bash
python experiment.py --model resnet34 --mode regularization --optimizer adam --regularizer dropout --values 0.0,0.2,0.4
```

## Быстрая проверка без долгого обучения

Чтобы не ждать вечность, как это обычно бывает с учебными заданиями на CPU, можно сначала запустить на подмножестве данных:

```bash
python experiment.py --model lenet --mode optimizers --epochs 2 --subset-train 2000 --subset-test 500
```

То же работает и для VGG16 / ResNet34.

## Полезные замечания

1. Датасеты скачиваются автоматически в папку `./data`.
2. Для `NAG` используется `torch.optim.SGD(momentum=0.9, nesterov=True)`.
3. Для LeNet вход MNIST преобразуется в формат `3x32x32`, чтобы соответствовать архитектурной схеме из задания.
4. Для CIFAR используются стандартные нормировки и базовые аугментации: случайный crop и горизонтальное отражение.
5. Результаты зависят от оборудования. На GPU серии экспериментов пройдут заметно быстрее, чем на CPU, спасибо человечеству за это редкое исключение.

## Что вставлять в отчет

После запуска серии экспериментов в отчет обычно вставляют:

- таблицы из `summary.csv`
- графики `val_loss_comparison.png`
- графики `val_accuracy_comparison.png`
- краткие выводы по скорости сходимости, итоговой точности и влиянию регуляризации
# taskCNN
