import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from digitsynth import synth
from pipeline import Hyper, Trainer

SIDE = 28
CKPT = Path("checkpoint.pt")


def _to_tensor(img):
    gray = img.convert("L").resize((SIDE, SIDE), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return torch.from_numpy(arr)[None, None, :, :]


def _probs(net, batch, device):
    with torch.no_grad():
        return torch.softmax(net(batch.to(device)), dim=1).cpu().numpy()


def cmd_train(args):
    hyper = Hyper(
        arch=args.arch,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        per_class=args.per_class,
    )
    Trainer(hyper, weights=CKPT).fit()


def cmd_predict(args):
    if not CKPT.exists():
        raise SystemExit(f"checkpoint not found: {CKPT}; run `python cli.py train` first")
    net, device, meta = Trainer.load(CKPT)
    print(f"loaded arch={meta.get('arch')} best_val={meta.get('best_val', 0) * 100:.2f}%")

    if args.demo:
        X, y = synth(per_class=5, seed=7)
        probs = _probs(net, torch.from_numpy(X), device)
        preds = probs.argmax(1)
        print(f"demo: {(preds == y).sum()}/{len(y)} correct")
        for t, p, conf in zip(y.tolist(), preds.tolist(), probs.max(1)):
            tag = "ok" if t == p else "!!"
            print(f"  {tag} true={t} pred={p} p={conf:.3f}")
        return

    if not args.images:
        raise SystemExit("nothing to predict: pass --demo or image paths")

    for path in args.images:
        probs = _probs(net, _to_tensor(Image.open(path)), device)[0]
        print(f"{path}: {int(probs.argmax())}  (p={probs.max():.3f})")


def main():
    parser = argparse.ArgumentParser(prog="digit-rec-v3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--arch", default="small_conv", choices=["small_conv"])
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--per-class", dest="per_class", type=int, default=300)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("images", nargs="*")
    p.add_argument("--demo", action="store_true")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()