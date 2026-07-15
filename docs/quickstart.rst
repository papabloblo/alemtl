Quickstart
==========

This page shows the minimum end-to-end ALE-MTL workflow: create a multitask
model, compute ALE profiles, and compare tasks from the resulting curves.

Model Layout
------------

``MultiTaskModel`` is built from named layers. Each layer is either hard-shared
across all tasks or soft-shared with one module per task.

.. code-block:: python

   import torch
   from torch import nn

   from alemtl.models import MultiTaskModel

   model = MultiTaskModel(
       n_tasks=3,
       modules_layout={
           "encoder": {
               "shared": "hard",
               "module": lambda: nn.Sequential(nn.Linear(2, 16), nn.ReLU()),
           },
           "head": {
               "shared": "soft",
               "module": lambda: nn.Linear(16, 1),
           },
       },
       similarity_layers={"in": "encoder", "out": "head"},
   )

   x = torch.randn(3, 32, 2)
   y = model(x)
   print(y.shape)  # torch.Size([3, 32, 1])

Task-First Batch Contract
-------------------------

ALE-MTL expects task-first tensors:

* input batches: ``(n_tasks, batch_size, n_features)``
* target batches: ``(n_tasks, batch_size, n_outputs)``
* model outputs: ``(n_tasks, batch_size, n_outputs)``

For synthetic examples, a dataloader can be represented as an iterable of
``(x, y)`` pairs:

.. code-block:: python

   batches = [
       (torch.randn(3, 64, 2), torch.randn(3, 64, 1))
       for _ in range(8)
   ]

ALE Profiles
------------

``MultiTaskALE`` computes local effects on the model slice configured by
``similarity_layers``.

.. code-block:: python

   from alemtl.similarity import MultiTaskALE

   ale = MultiTaskALE(
       model=model,
       dataloader=batches,
       n_tasks=3,
       n_features_out=1,
       num_intervals=10,
   )
   ale.update()

   curves = ale(centered=True, cumulative=True, std=1.0)
   print(curves.shape)

Task Similarity
---------------

``MultitaskSimilarity`` compares ALE curves and returns pairwise task
similarity scores and nearest-task groups.

.. code-block:: python

   from alemtl.similarity import MultitaskSimilarity

   similarity = MultitaskSimilarity(ale)
   similarity.compute()
   scores, task_pairs = similarity.tasks_groups()

   print(similarity.scores)
   print(scores)
   print(task_pairs)
