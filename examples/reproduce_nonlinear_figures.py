"""Reproduce two interpretation figures for the table's nonlinear dataset.

Run from the repository root:
    python examples/reproduce_nonlinear_figures.py --output results/nonlinear_figures

Requires matplotlib and the adjacent reproduce_revised_table.py. Uses the same
split (1101), validation seed (1201), final seed (1301), architecture, scaling,
regularization search and training budget as the table. Recomputes explanations
on training inputs after restoring the best-validation checkpoint. Test data
are never used for configuration or figure selection.
"""
from pathlib import Path
import argparse
from contextlib import redirect_stdout
from io import StringIO
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reproduce_revised_table import (
    Panel, Batches, build_model, prepare_panel, run_one, initialize, STRENGTHS,
    MultiTaskALE, MultitaskSimilarity, MultiTaskLoss, MultiTaskTrainer, rmse_loss,
)

def fit_for_figures(panel, method, seed, strength, epochs):
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
    # Explicitly recompute explanations after the trainer restores the best model.
    ale.recompute()
    similarity.compute()
    curves = ale(centered=True, cumulative=True, std=1.0).detach().cpu().numpy()
    matrix = similarity.similarity_tasks_features.mean(dim=-1).detach().cpu().numpy()
    weights, pairs = similarity.tasks_groups()
    return model, curves, matrix, pairs.detach().cpu().numpy(), trainer.tracking.best_epoch

def plot_figures(curves, matrix, pairs, out):
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    colors = ['#1b6ca8', '#1b6ca8', '#b45309', '#b45309', '#23834b', '#23834b']
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), layout='constrained')
    for feature, ax in enumerate(axes.flat[:5]):
        for task in range(6):
            ax.plot(curves[task,feature,:,0], curves[task,feature,:,1],
                    color=colors[task], ls='-' if task % 2 == 0 else '--',
                    lw=1.7, label=f'Task {task+1}')
        ax.set(title=f'Input feature {feature+1}', xlabel='Standardized input',
               ylabel='Normalized ALE')
        ax.grid(alpha=.18)
    axes.flat[5].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc='upper left', ncol=2, frameon=False)
    axes.flat[5].text(0, .4, 'Known task pairs:\n(1, 2), (3, 4), (5, 6)\n\nColors identify generating pairs;\ncurves come from the fitted model.',
                      transform=axes.flat[5].transAxes, va='top', linespacing=1.5)
    fig.savefig(out/'nonlinear_ale_profiles.pdf', bbox_inches='tight');plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained',
                            gridspec_kw={'width_ratios':[1.05,1,1]})
    masked=np.ma.array(matrix,mask=np.eye(6,dtype=bool))
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
    im=axes[0].imshow(masked,vmin=0,vmax=1,cmap=cmap)
    for i in range(6):
        for j in range(6):
            if i!=j: axes[0].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,
                                  color='black' if matrix[i,j]>.6 else 'white')
    axes[0].set(xticks=range(6),yticks=range(6),xticklabels=range(1,7),yticklabels=range(1,7),
                xlabel='Task',ylabel='Task',title='Mean feature similarity')
    fig.colorbar(im,ax=axes[0],shrink=.75,pad=.03)
    values=matrix[np.triu_indices(6,1)]
    axes[1].hist(values,bins=np.linspace(0,1,9),color='#1b6ca8',edgecolor='white')
    axes[1].set(xlabel='Mean feature similarity',ylabel='Number of unique pairs',
                title='All 15 task pairs',xlim=(0,1))
    pos=np.array([[-.55,.85],[.55,.85],[-.55,0],[.55,0],[-.55,-.85],[.55,-.85]])
    for i,j in pairs:
        axes[2].annotate('',xy=pos[j],xytext=pos[i],arrowprops=dict(
            arrowstyle='-|>',color='#555555',lw=1+2*matrix[i,j],shrinkA=16,shrinkB=16,
            connectionstyle='arc3,rad=0.12'))
    for i,xy in enumerate(pos):
        axes[2].scatter(*xy,s=600,color=colors[i],zorder=3)
        axes[2].text(*xy,str(i+1),ha='center',va='center',color='white',weight='bold',zorder=4)
    axes[2].set(title='Inferred nearest neighbours',xlim=(-1,1),ylim=(-1.2,1.2));axes[2].axis('off')
    fig.savefig(out/'nonlinear_similarity_structure.pdf',bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/nonlinear_figures'))
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()): parser.error('Choose an empty output directory')
    initialize()
    panel=prepare_panel('clustered_nonlinear',1101)
    validation=Panel({**panel.splits,'test':panel.splits['validation']},panel.y_mean,panel.y_scale)
    scores={str(s):run_one(validation,'ale_frechet',1201,s,120) for s in STRENGTHS}
    strength=min(STRENGTHS,key=lambda s:scores[str(s)])
    # Even the figure fit uses validation in the trainer evaluation slot.
    model,curves,matrix,pairs,epoch=fit_for_figures(validation,'ale_frechet',1301,strength,120)
    out.mkdir(parents=True,exist_ok=True)
    plot_figures(curves,matrix,pairs,out)
    np.savez_compressed(out/'explanations.npz',curves=curves,mean_similarity=matrix,pairs=pairs)
    torch.save(model.state_dict(),out/'checkpoint.pt')
    sources=[Path(__file__),Path(__file__).with_name('reproduce_revised_table.py')]
    manifest=dict(dataset='clustered_nonlinear',data_seed=271829,split_seed=1101,
        validation_seed=1201,initialization_seed=1301,epochs=120,best_epoch_zero_based=int(epoch),
        regularization=strength,validation_scores=scores,width=32,activation='tanh',
        learning_rate=.003,batch_size=64,intervals=8,update_every=5,curve_std=1.0,
        explanation_source='Fresh recomputation on training inputs at restored best-validation checkpoint',
        display_similarity='Mean across five input features; training library scores sum features',
        peers_one_based=(pairs[:,1]+1).tolist(),pair_accuracy=float(np.mean(pairs[:,1]==(np.arange(6)^1))),
        python=platform.python_version(),versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','matplotlib']},
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
