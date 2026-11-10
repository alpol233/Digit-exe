# Digit recognizer v3

Свёрточная сеть для цифр 0–9 (28×28, grayscale).

## Файлы
- `digitsynth.py` — синтез датасета
- `networks.py`   — фабрика моделей (`build`)
- `pipeline.py`   — класс `Trainer`, чекпойнт `checkpoint.pt`
- `cli.py`        — подкоманды `train` / `predict`

## Использование
    pip install -r requirements.txt
    python cli.py train
    python cli.py predict --demo
    python cli.py predict file1.png file2.png