"""Reproduce the comparison and write only a LaTeX table (requires booktabs).

    python examples/reproduce_revised_table.py --output results/table.tex

Five offline datasets, five splits, and two final seeds. Regularization is
selected on validation data before test evaluation. Reported SD is across split
means; overlapping splits and adapted regression tasks are exploratory evidence.
Use --smoke for a short pipeline check, not publication results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_wine, load_breast_cancer

_SOURCE = Path(__file__).resolve().parents[1] / "src"
if (_SOURCE / "alemtl").is_dir():
    sys.path.insert(0, str(_SOURCE))

from alemtl.models import MultiTaskModel
from alemtl.similarity import MultiTaskALE, MultitaskSimilarity
from alemtl.training import MultiTaskLoss, MultiTaskTrainer, rmse_loss

DATASETS = {
    "clustered_sparse": "Linear, 80 samples/task", "clustered_dense": "Linear, 240 samples/task",
    "clustered_nonlinear": "Nonlinear", "wine_malic_acid": "Wine malic acid",
    "breast_radius": "Breast radius",
}
METHODS = {"single": "Independent", "hard": "Hard sharing",
           "soft": "Soft sharing", "ale_frechet": "ALE default"}
SPLITS = [1101, 1102, 1103, 1104, 1105]
STRENGTHS = [0., .01, .1, 1., 10.]

@dataclass
class Panel:
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]]
    y_mean: torch.Tensor
    y_scale: torch.Tensor


class Batches:
    def __init__(self, tensors, size, seed, shuffle=False):
        self.x, self.y = tensors
        self.size, self.shuffle = size, shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (self.x.size(1) + self.size - 1) // self.size

    def __iter__(self):
        order = torch.randperm(self.x.size(1), generator=self.generator) if self.shuffle else torch.arange(self.x.size(1))
        for start in range(0, len(order), self.size):
            ids = order[start:start+self.size]
            yield self.x[:, ids], self.y[:, ids]


def build_model(method, tasks, features):
    return MultiTaskModel(tasks, {
        "encoder": {"shared": "hard" if method == "hard" else "soft",
                    "module": lambda: nn.Sequential(nn.Linear(features, 32), nn.Tanh())},
        "head": {"shared": "soft", "module": lambda: nn.Linear(32, 1)},
    }, similarity_layers={"in": "encoder", "out": "head"}, same_parameters=True)


def run_one(panel, method, seed, strength, epochs):
    torch.manual_seed(seed)
    tasks, _, features = panel.splits["train"][0].shape
    model = build_model(method, tasks, features)
    loaders = {key: Batches(tensors, 64, seed, key == "train") for key, tensors in panel.splits.items()}
    ale = similarity = None
    coefficient = 0.0
    if method in ("ale_frechet", "ale_unscaled"):
        ale = MultiTaskALE(model, Batches(panel.splits["train"], 64, seed),
                           n_tasks=tasks, num_intervals=8,
                           n_guess=panel.splits["train"][0].size(1))
        similarity = MultitaskSimilarity(ale, std=None if method == "ale_unscaled" else 1.0)
        coefficient = strength / (tasks * features)
    pairs = [[i, j] for i in range(tasks) for j in range(i + 1, tasks)]
    if method == "soft":
        coefficient = strength / len(pairs)
    loss = MultiTaskLoss(model, nn.MSELoss(reduction="none"),
                         errors_fn={"rmse": rmse_loss}, l2_penalty=coefficient)
    if method == "soft":
        loss.update_tasks_groups(pairs)
    trainer = MultiTaskTrainer(
        model, loaders["train"], loaders["validation"], loaders["test"],
        torch.optim.Adam(model.parameters(), lr=.003), loss,
        ale=ale, multitask_similarity=similarity,
        ale_each_epochs=5 if ale else None,
        similarity_each_epochs=5 if ale else None,
        keep_similarity_epochs=5, early_stopping_epochs=epochs + 1,
        print_each_epochs=epochs, print_limit_epochs=1, logging_dir="",
    )
    with redirect_stdout(StringIO()):
        trainer.train(epochs=epochs)
    # The trainer restores the validation-selected checkpoint before testing.
    with torch.no_grad():
        x, y = panel.splits["test"]
        prediction = model(x) * panel.y_scale + panel.y_mean
        target = y * panel.y_scale + panel.y_mean
        per_task = (prediction-target).square().mean(dim=(1, 2)).sqrt()
    if not torch.isfinite(per_task).all():
        raise RuntimeError(f"Non-finite RMSE for {method}, seed {seed}")
    return per_task.mean().item()


def prepare_panel(name, split_seed):
    rng=np.random.default_rng(271829)
    groups=[]
    if name == 'wine_malic_acid':
        data=load_wine()
        for task in np.unique(data.target):
            ids=np.flatnonzero(data.target==task)
            groups.append((np.delete(data.data[ids],1,axis=1),data.data[ids,1]))
    elif name == 'breast_radius':
        data=load_breast_cancer()
        # Texture, smoothness, compactness, concavity, concave points, symmetry,
        # fractal dimension; omit all radius, area and perimeter measurements.
        columns=[1,4,5,6,7,8,9]
        for task in np.unique(data.target):
            ids=np.flatnonzero(data.target==task)
            groups.append((data.data[ids][:,columns],data.data[ids,0]))
    else:
        n=240 if name=='clustered_dense' else 80
        centers=rng.normal(size=(3,5))
        coefficients=np.repeat(centers,2,axis=0)+rng.normal(0,.05,(6,5))
        for coef in coefficients:
            x=rng.uniform(-1,1,(n,5))
            basis=np.column_stack([np.sin(2*x[:,0]),x[:,1]**2,x[:,2],np.tanh(2*x[:,3]),x[:,4]]) if name=='clustered_nonlinear' else x
            y=basis@coef+rng.normal(0,.5,n)
            groups.append((x,y))
    rng=np.random.default_rng(split_seed)
    parts={key:[] for key in ('train','validation','test')}
    for x,y in groups:
        order=rng.permutation(len(y))
        for key,indices in zip(parts,np.split(order,[int(.6*len(y)),int(.8*len(y))])):
            parts[key].append((x[indices],y[indices,None]))
    sizes={key:min(len(y) for x,y in entries) for key,entries in parts.items()}
    arrays={key:(np.stack([x[:sizes[key]] for x,y in entries]),np.stack([y[:sizes[key]] for x,y in entries])) for key,entries in parts.items()}
    tx,ty=arrays['train']
    xm,xs=tx.mean(1,keepdims=True),np.maximum(tx.std(1,keepdims=True),1e-8)
    ym,ys=ty.mean(1,keepdims=True),np.maximum(ty.std(1,keepdims=True),1e-8)
    splits={key:(torch.tensor((x-xm)/xs,dtype=torch.float32),torch.tensor((y-ym)/ys,dtype=torch.float32)) for key,(x,y) in arrays.items()}
    return Panel(splits,torch.tensor(ym,dtype=torch.float32),torch.tensor(ys,dtype=torch.float32))




def initialize():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def select(job):
    dataset, split, epochs, strengths, _ = job
    panel = prepare_panel(dataset, split)
    # The trainer's evaluation loader receives validation data during selection.
    validation = Panel({**panel.splits, "test": panel.splits["validation"]},
                       panel.y_mean, panel.y_scale)
    selected = {}
    for method in METHODS:
        candidates = [0.] if method in ("single", "hard") else strengths
        selected[method] = min(candidates, key=lambda strength:
            run_one(validation, method, 1201, strength, epochs))
    print(f"{dataset}, split {split}: validation complete", flush=True)
    return job, selected


def evaluate(selection):
    (dataset, split, epochs, _, seeds), selected = selection
    panel = prepare_panel(dataset, split)
    scores = [np.mean([run_one(panel, method, seed, selected[method], epochs)
                       for seed in seeds]) for method in METHODS]
    print(f"{dataset}, split {split}: test complete", flush=True)
    return dataset, scores


def write_table(results, output, smoke):
    lines = [r"% Requires \usepackage{booktabs}", r"\begin{table*}[t]",
             r"\centering", r"\small",
             r"\begin{tabular}{l" + "c" * len(METHODS) + "}", r"\toprule",
             "Dataset & " + " & ".join(METHODS.values()) + r" \\", r"\midrule"]
    for dataset, label in DATASETS.items():
        scores = np.array([scores for name, scores in results if name == dataset])
        if not scores.size:
            continue
        means, deviations = scores.mean(axis=0), scores.std(axis=0, ddof=1)
        cells = []
        for mean, deviation in zip(means, deviations):
            cell = f"{mean:.5f} \\pm {deviation:.5f}"
            if mean == means.min():
                cell = r"\mathbf{" + cell + "}"
            cells.append("$" + cell + "$")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    caption = ("SMOKE TEST ONLY; not publication results." if smoke else
               r"Test macro-RMSE (lower is better): mean $\pm$ sample SD across five "
               "split means, each averaging two initialization seeds. Configuration "
               "and checkpoint selection use validation data only. Bold denotes the "
               "lowest mean, not statistical significance. Synthetic tasks and adapted "
               "real-data regression tasks are exploratory; splits overlap.")
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{" + caption + "}",
              r"\label{tab:compact-revised}", r"\end{table*}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/table.tex"),
                        help="LaTeX output file (default: results/table.tex)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    if options.workers < 1:
        parser.error("--workers must be positive")
    if options.output.suffix != ".tex":
        parser.error("--output must be a .tex file")
    if options.output.exists():
        parser.error("Output already exists; choose a new .tex file")
    datasets = ["clustered_sparse", "wine_malic_acid"] if options.smoke else DATASETS
    splits = SPLITS[:2] if options.smoke else SPLITS
    jobs = [(dataset, split, 6 if options.smoke else 120,
             [0., .1] if options.smoke else STRENGTHS,
             [1301] if options.smoke else [1301, 1302])
            for dataset in datasets for split in splits]
    with ProcessPoolExecutor(max_workers=options.workers, initializer=initialize) as pool:
        selected = list(pool.map(select, jobs))
        print("All configurations selected. Starting test evaluation.", flush=True)
        results = list(pool.map(evaluate, selected))
    write_table(results, options.output, options.smoke)
    print(f"Table saved to {options.output}")


if __name__ == "__main__":
    main()
