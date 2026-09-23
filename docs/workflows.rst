Core Workflows
==============

ALE-MTL is organized around four repeatable workflows.

Build a Multitask Model
-----------------------

Use :class:`alemtl.models.MultiTaskModel` when you want explicit control over
which layers are shared. Hard-shared layers broadcast one module over all tasks.
Soft-shared layers keep one module per task and expose task-specific parameters
for regularization.

Use the baseline models in :mod:`alemtl.models.baselines` when comparing against
standard multitask architectures such as hard sharing, soft sharing, MMoE,
Cross-Stitch, PLE, or MTAN-style tabular models.

Load Task-First Data
--------------------

The data utilities convert common tabular or image datasets into task-first
batches. ``MultitaskDataloader`` wraps PyTorch dataloaders and moves nested
tensors to the configured device while preserving the ALE-MTL batch contract.

Compute ALE Profiles
--------------------

``MultiTaskALE`` samples feature intervals, perturbs features interval by
interval, accumulates local effects, and returns task-wise ALE curves. These
curves can be centered, accumulated, standardized, or smoothed depending on the
comparison workflow.

Use ``ale.recompute()`` to explain the current model after training or changing
its parameters. This rebuilds intervals in the current input or latent feature
space, clears previous effects and counts, and computes fresh curves in
evaluation mode. The original module training modes are restored afterward.
``ale.recompute(max_batches=2)`` limits the effect computation for a preview;
interval initialization still samples according to ``n_guess``.

``ale.update()`` remains an incremental operation for an unchanged model.
``ale.reset()`` rebuilds intervals and clears accumulators without computing
effects. Scheduled trainer updates use ``recompute()`` so task similarities do
not mix explanations from different training epochs. Previously generated
training results that used accumulated historical effects must be regenerated
to reflect this behavior.

Compare Tasks and Train
-----------------------

``MultitaskSimilarity`` converts ALE curves into task-distance and
task-similarity matrices. ``MultiTaskTrainer`` can schedule ALE recomputation
during training and pass nearest-task groups to ``MultiTaskLoss`` for
similarity-aware soft-sharing regularization.

Recommended Reproducibility Pattern
-----------------------------------

For SoftwareX-style artifacts, each experiment should document:

* package version and Git commit;
* random seeds;
* dataset source or synthetic data generator;
* model layout;
* ALE interval count and similarity metric;
* training loss, optimizer, and scheduler;
* commands used to regenerate outputs.