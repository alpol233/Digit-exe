import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_SIZE = 28
WORK_SIZE = 56

GLYPHS = {
    0: [(14, 8, 40, 8), (40, 8, 40, 46), (40, 46, 14, 46), (14, 46, 14, 8)],
    1: [(28, 8, 28, 46)],
    2: [(14, 10, 40, 10), (40, 10, 40, 26), (40, 26, 14, 26),
        (14, 26, 14, 44), (14, 44, 40, 44)],
    3: [(14, 10, 40, 10), (40, 10, 40, 44), (40, 44, 14, 44), (20, 27, 40, 27)],
    4: [(14, 10, 14, 30), (14, 30, 40, 30), (40, 10, 40, 46)],
    5: [(40, 10, 14, 10), (14, 10, 14, 27), (14, 27, 40, 27),
        (40, 27, 40, 44), (40, 44, 14, 44)],
    6: [(40, 10, 16, 10), (16, 10, 16, 44), (16, 44, 40, 44),
        (40, 44, 40, 27), (40, 27, 16, 27)],
    7: [(14, 10, 40, 10), (40, 10, 20, 46)],
    8: [(14, 10, 40, 10), (40, 10, 40, 44), (40, 44, 14, 44),
        (14, 44, 14, 10), (14, 27, 40, 27)],
    9: [(40, 44, 40, 12), (40, 12, 16, 12), (16, 12, 16, 27),
        (16, 27, 40, 27), (16, 12, 16, 10)],
}


def _draw(digit, rng):
    canvas = Image.new("L", (WORK_SIZE, WORK_SIZE), 0)
    brush = ImageDraw.Draw(canvas)
    stroke = int(rng.integers(2, 5))
    for x1, y1, x2, y2 in GLYPHS[digit]:
        brush.line((x1, y1, x2, y2), fill=255, width=stroke)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.0))))
    canvas = canvas.rotate(float(rng.uniform(-12, 12)), resample=Image.BILINEAR, fillcolor=0)
    ox, oy = (int(v) for v in rng.integers(-3, 4, size=2))
    return Image.fromarray(np.roll(np.array(canvas), (oy, ox), axis=(0, 1)))


def _normalize(canvas, rng):
    raw = np.asarray(canvas, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(raw > 0.1)
    if len(xs) == 0:
        return np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)
    crop = raw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    thumbnail = Image.fromarray((crop * 255).astype(np.uint8))
    thumbnail.thumbnail((20, 20), Image.BILINEAR)
    plate = Image.new("L", (OUT_SIZE, OUT_SIZE), 0)
    plate.paste(thumbnail, ((OUT_SIZE - thumbnail.width) // 2, (OUT_SIZE - thumbnail.height) // 2))
    img = np.asarray(plate, dtype=np.float32) / 255.0
    return np.clip(img + rng.normal(0, 0.03, img.shape).astype(np.float32), 0.0, 1.0)


def synth(per_class=300, seed=42):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for d in range(10):
        for _ in range(per_class):
            xs.append(_normalize(_draw(d, rng), rng))
            ys.append(d)
    X = np.stack(xs).astype(np.float32)[:, None, :, :]
    y = np.asarray(ys, dtype=np.int64)
    return X, y