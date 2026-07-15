"""Composable parameter-sharing model for multi-task learning.

The model in this module follows the project-wide tensor contract:

* regular multitask input: ``(n_tasks, batch, ...)``
* shared input: ``(batch, ...)`` when ``shared_input_data=True``
* output: ``(n_tasks, batch, ...)``

Each configured layer is wrapped either as a hard-shared module, where one
``nn.Module`` instance is reused for all tasks, or as a task-specific module
group, where each task owns its own module instance.
"""

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, TypedDict

import torch
from torch import nn

ModuleFactory = Callable[[], nn.Module]


class ModuleSpec(TypedDict):
    """Configuration for one layer in :class:`MultiTaskModel`.

    ``shared`` controls how the layer is wrapped:

    * ``"hard"``: one module instance is reused for every task.
    * ``"soft"``: one module instance is created per task and included in the
      soft-sharing parameter utilities.
    * ``None``: one module instance is created per task, but the layer is
      treated as task-specific rather than soft-shared for parameter grouping.
    """

    shared: Optional[str]
    module: ModuleFactory


Layer = Tuple[str, ModuleSpec]
LayoutInput = Union[Mapping[str, ModuleSpec], Sequence[Layer]]


# ---------- Parameter initialization ----------


def _init_weight_(tensor: torch.Tensor, weight_init: str, fallback_range: float) -> None:
    """Initialize a trainable non-bias tensor in place."""

    if tensor.dim() < 2:
        nn.init.uniform_(tensor, -fallback_range, fallback_range)
        return

    if weight_init == "kaiming_uniform":
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu")
    elif weight_init == "xavier_uniform":
        nn.init.xavier_uniform_(tensor)
    elif weight_init == "normal_0_02":
        nn.init.normal_(tensor, mean=0.0, std=0.02)
    else:
        raise ValueError(
            "Unknown weight_init "
            f"{weight_init!r}; expected 'kaiming_uniform', 'xavier_uniform', or 'normal_0_02'."
        )


def initialize_module_parameters_(
    module: nn.Module,
    *,
    weight_init: str = "kaiming_uniform",
    bias_range: float = 0.1,
) -> None:
    """Initialize trainable parameters without touching buffers.

    This is used when task-specific modules need identical starting values.
    Buffers such as BatchNorm running statistics are preserved from the module
    factory instead of being randomly overwritten.
    """

    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if "bias" in name:
            nn.init.uniform_(parameter, -bias_range, bias_range)
        else:
            _init_weight_(parameter, weight_init, bias_range)


def clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a detached tensor clone of ``module.state_dict()``."""

    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


# ---------- Hard sharing ----------


class SharedModule(nn.Module):
    """Apply one module instance to all tasks.

    Parameters
    ----------
    module:
        Module shared by every task.
    n_tasks:
        Number of task outputs to produce.
    collapse_shared_input:
        If ``True``, a raw shared input batch ``(batch, ...)`` is evaluated
        once and broadcast to ``(n_tasks, batch, ...)``. A zero-stride
        broadcast from a previous shared layer is collapsed back to one task
        before evaluation.
    """

    def __init__(
        self,
        module: nn.Module,
        n_tasks: int,
        collapse_shared_input: bool = False,
    ) -> None:
        super().__init__()
        self.module = module
        self.n_tasks = int(n_tasks)
        self.collapse_shared_input = bool(collapse_shared_input)

    def _broadcast(self, y: torch.Tensor) -> torch.Tensor:
        return y.unsqueeze(0).expand(self.n_tasks, *y.shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Run the shared module and return a task-indexed tensor."""

        if not self.collapse_shared_input:
            if X.size(0) != self.n_tasks:
                raise ValueError(f"Expected X.size(0) == n_tasks ({self.n_tasks}), got {X.size(0)}.")
            return torch.stack([self.module(x) for x in X], dim=0)

        if X.dim() >= 1 and X.size(0) == 1:
            return self._broadcast(self.module(X[0]))

        if X.dim() >= 1 and X.size(0) == self.n_tasks and X.stride(0) == 0:
            return self._broadcast(self.module(X[0]))

        return self._broadcast(self.module(X))


# ---------- Task-specific / soft sharing ----------


class SoftSharedModule(nn.Module):
    """Keep one module instance per task.

    When ``same_parameters=True``, the first task module is initialized once
    using ``weight_init``/``bias_range`` and its full state dict is copied into
    every task module. The task modules remain independent after construction.
    """

    def __init__(
        self,
        n_tasks: int,
        module_factory: ModuleFactory,
        *,
        same_parameters: bool = False,
        weight_init: str = "kaiming_uniform",
        bias_range: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.task_nets = nn.ModuleList([module_factory() for _ in range(self.n_tasks)])

        if same_parameters and self.task_nets:
            initialize_module_parameters_(
                self.task_nets[0],
                weight_init=weight_init,
                bias_range=bias_range,
            )
            template_state = clone_state_dict(self.task_nets[0])
            for net in self.task_nets:
                net.load_state_dict(template_state)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply each task-specific module to its corresponding task input."""

        if X.size(0) != self.n_tasks:
            raise ValueError(f"Expected X.size(0) == n_tasks ({self.n_tasks}), got {X.size(0)}.")
        return torch.stack([self.task_nets[i](X[i]) for i in range(self.n_tasks)], dim=0)


# ---------- Multi-task model ----------


class MultiTaskModel(nn.Module):
    """Composable multitask model with hard-shared and task-specific layers.

    ``modules_layout`` defines the execution order. It can be either an
    insertion-ordered mapping or a sequence of ``(layer_name, ModuleSpec)``
    tuples. Each factory must return a fresh ``nn.Module`` instance.

    ``similarity_layers`` defines a named inclusive slice of the model used by
    the ALE/similarity code. For example, ``{"in": "trunk", "out": "head"}``
    creates:

    * ``model_similarity_input``: all layers before ``"trunk"``.
    * ``model_similarity``: layers from ``"trunk"`` through ``"head"``.

    The submodels are shallow views over the same module instances, not deep
    copies. Updating the main model updates the submodels.
    """

    def __init__(
        self,
        n_tasks: int,
        modules_layout: LayoutInput,
        similarity_layers: Mapping[str, Optional[str]],
        *,
        device: Union[str, torch.device] = "cpu",
        same_parameters: bool = False,
        shared_input_data: bool = False,
        weight_init: str = "kaiming_uniform",
        bias_range: float = 0.1,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.n_tasks = int(n_tasks)
        if self.n_tasks <= 0:
            raise ValueError(f"n_tasks must be positive, got {n_tasks!r}.")

        self.shared_input_data = bool(shared_input_data)
        self.modules_layout = self._build_modules_layout(modules_layout)
        self.similarity_layers = self._validate_similarity_layers(similarity_layers)

        self.model = self._create_model(
            same_parameters=same_parameters,
            weight_init=weight_init,
            bias_range=bias_range,
        ).to(self.device)

        self.model_similarity = self.create_submodel(self.get_name_similarity_layers())
        self.model_similarity_input = self.create_submodel(self.get_name_input_similarity_layers())

    # ---- builders ----

    @staticmethod
    def _layout_items(layers: LayoutInput) -> List[Layer]:
        if isinstance(layers, Mapping):
            return list(layers.items())
        return list(layers)

    @classmethod
    def _build_modules_layout(cls, layers: LayoutInput) -> Dict[str, ModuleSpec]:
        """Validate and normalize the layer layout while preserving order."""

        items = cls._layout_items(layers)
        if not items:
            raise ValueError("modules_layout must contain at least one layer.")

        normalized: Dict[str, ModuleSpec] = {}
        for item in items:
            try:
                name, spec = item
            except (TypeError, ValueError) as exc:
                raise ValueError("Each modules_layout entry must be a (name, spec) pair.") from exc

            if not isinstance(name, str) or not name:
                raise ValueError(f"Layer names must be non-empty strings, got {name!r}.")
            if name in normalized:
                raise ValueError(f"Duplicate layer name {name!r}.")
            if not isinstance(spec, Mapping):
                raise ValueError(f"Layer {name!r} spec must be a mapping, got {type(spec).__name__}.")
            if "module" not in spec:
                raise ValueError(f"Layer {name!r} is missing required 'module' factory.")

            factory = spec["module"]
            if not callable(factory):
                raise ValueError(f"Layer {name!r} module factory must be callable.")

            shared_type = spec.get("shared")
            if shared_type not in ("hard", "soft", None):
                raise ValueError(f"Unknown shared type {shared_type!r} for layer {name!r}.")

            normalized[name] = {"shared": shared_type, "module": factory}

        return normalized

    def _validate_similarity_layers(
        self,
        similarity_layers: Mapping[str, Optional[str]],
    ) -> Dict[str, Optional[str]]:
        """Validate the similarity slice configuration."""

        out_name = similarity_layers.get("out")
        if out_name is None:
            raise ValueError("similarity_layers['out'] must be provided.")

        layer_names = self._all_layer_names()
        if out_name not in layer_names:
            raise ValueError(f"similarity_layers['out']={out_name!r} is not a configured layer.")

        in_name = similarity_layers.get("in")
        if in_name is not None and in_name not in layer_names:
            raise ValueError(f"similarity_layers['in']={in_name!r} is not a configured layer.")

        if in_name is not None and layer_names.index(in_name) > layer_names.index(out_name):
            raise ValueError(f"Invalid similarity layer range: in={in_name!r}, out={out_name!r}.")

        return {"in": in_name, "out": out_name}

    def _create_model(
        self,
        *,
        same_parameters: bool,
        weight_init: str,
        bias_range: float,
    ) -> nn.Sequential:
        model = nn.Sequential()
        task_specific_seen = False

        for module_name, spec in self.modules_layout.items():
            shared_type = spec["shared"]
            factory = spec["module"]

            if shared_type == "hard":
                module_instance = SharedModule(
                    module=factory(),
                    n_tasks=self.n_tasks,
                    collapse_shared_input=self.shared_input_data and not task_specific_seen,
                )
            else:
                task_specific_seen = True
                module_instance = SoftSharedModule(
                    n_tasks=self.n_tasks,
                    module_factory=factory,
                    same_parameters=same_parameters,
                    weight_init=weight_init,
                    bias_range=bias_range,
                )

            model.add_module(name=module_name, module=module_instance)

        return model

    def create_submodel(
        self,
        layer_names: Optional[Sequence[str]],
        model_input: Optional[nn.Sequential] = None,
    ) -> nn.Sequential:
        """Create a shallow sequential view over named layers.

        ``layer_names=None`` returns an identity submodel. This represents the
        empty prefix before the similarity slice when the slice starts at the
        model input.
        """

        source_model = self.model if model_input is None else model_input
        if layer_names is None:
            return nn.Sequential(nn.Identity())

        submodel = nn.Sequential()
        for name in layer_names:
            submodel.add_module(name, source_model.get_submodule(name))
        return submodel

    # ---- layer name helpers ----

    def _all_layer_names(self) -> List[str]:
        return list(self.modules_layout.keys())

    def get_name_task_specific_layers(self) -> List[str]:
        """Return names for layers that have per-task module instances."""

        return [name for name, spec in self.modules_layout.items() if spec["shared"] != "hard"]

    def get_name_soft_shared_layers(self) -> List[str]:
        """Return names for explicit soft-shared layers.

        Layers configured with ``shared=None`` are task-specific but omitted
        from this list so regularizers can target only explicit soft-sharing
        layers.
        """

        return [name for name, spec in self.modules_layout.items() if spec["shared"] == "soft"]

    def get_name_similarity_layers(self) -> List[str]:
        """Return the inclusive layer-name slice used for similarity."""

        layers = self._all_layer_names()
        in_name = self.similarity_layers["in"]
        out_name = self.similarity_layers["out"]
        in_idx = layers.index(in_name) if in_name is not None else 0
        out_idx = layers.index(out_name)
        return layers[in_idx : out_idx + 1]

    def get_name_input_similarity_layers(self) -> Optional[List[str]]:
        """Return layer names before the similarity slice, or ``None``."""

        in_name = self.similarity_layers["in"]
        if in_name is None:
            return None

        layers = self._all_layer_names()
        in_idx = layers.index(in_name)
        return layers[:in_idx] if in_idx > 0 else None

    # ---- parameter utilities ----

    @staticmethod
    def _empty_parameter_vector(module: nn.Module) -> torch.Tensor:
        parameter = next(module.parameters(), None)
        if parameter is None:
            return torch.empty(0)
        return torch.empty(0, device=parameter.device, dtype=parameter.dtype)

    @classmethod
    def _flatten_weights_from(cls, module: nn.Module) -> torch.Tensor:
        parts = [p.flatten() for name, p in module.named_parameters() if "weight" in name]
        return torch.cat(parts, dim=0) if parts else cls._empty_parameter_vector(module)

    def _validate_task_index(self, task: int) -> int:
        task = int(task)
        if not 0 <= task < self.n_tasks:
            raise IndexError(f"task index must be in [0, {self.n_tasks}), got {task}.")
        return task

    def get_soft_shared_parameters_by_task(self, task: int) -> torch.Tensor:
        """Concatenate explicit soft-shared weights for one task."""

        task = self._validate_task_index(task)
        tensors: List[torch.Tensor] = []
        for layer_name in self.get_name_soft_shared_layers():
            layer = self.model.get_submodule(layer_name)
            if not isinstance(layer, SoftSharedModule):
                raise TypeError(f"Layer {layer_name!r} is not a SoftSharedModule.")
            tensors.append(self._flatten_weights_from(layer.task_nets[task]))

        return torch.cat(tensors, dim=0) if tensors else torch.empty(0, device=self.device)

    def get_param_groups(self, tasks_groups: List[List[int]]) -> List[torch.Tensor]:
        """Return grouped soft-shared parameter vectors.

        Each returned tensor has shape ``(group_size, n_params)`` for the
        corresponding task group.
        """

        return [
            torch.stack([self.get_soft_shared_parameters_by_task(task) for task in group], dim=0)
            for group in tasks_groups
        ]

    # ---------- forwards ----------

    def _forward(self, X: torch.Tensor, model: nn.Module) -> torch.Tensor:
        return model(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Run the complete multitask model."""

        return self._forward(X, self.model)

    def forward_input_similarity(self, X: torch.Tensor) -> torch.Tensor:
        """Run layers before the similarity slice."""

        return self._forward(X, self.model_similarity_input)

    def forward_similarity(self, X: torch.Tensor) -> torch.Tensor:
        """Run the configured similarity slice."""

        return self._forward(X, self.model_similarity)

    # ---- per-task extractor ----

    def model_by_task(self, task: int) -> nn.ModuleDict:
        """Return per-task views of the full model and similarity submodels.

        The returned modules reference the same underlying parameters as the
        multitask model. They are intended for single-task inference or
        analysis, not for creating an independent copy.
        """

        task = self._validate_task_index(task)
        model = nn.Sequential()

        for module_name, spec in self.modules_layout.items():
            submodule = self.model.get_submodule(module_name)
            if spec["shared"] == "hard":
                if not isinstance(submodule, SharedModule):
                    raise TypeError(f"Layer {module_name!r} is not a SharedModule.")
                model.add_module(module_name, submodule.module)
            else:
                if not isinstance(submodule, SoftSharedModule):
                    raise TypeError(f"Layer {module_name!r} is not a SoftSharedModule.")
                model.add_module(module_name, submodule.task_nets[task])

        return nn.ModuleDict(
            {
                "model": model,
                "model_similarity": self.create_submodel(self.get_name_similarity_layers(), model),
                "model_input_similarity": self.create_submodel(
                    self.get_name_input_similarity_layers(),
                    model,
                ),
            }
        )

    @staticmethod
    def guess_output_shape(module: nn.Module, X: torch.Tensor) -> Tuple[torch.Size, torch.Tensor]:
        """Run ``module`` once and return both output shape and output tensor."""

        y = module(X)
        return y.shape, y

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Move the model and keep the public ``device`` attribute in sync."""

        out = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str)):
            self.device = torch.device(args[0])
        elif kwargs.get("device") is not None:
            self.device = torch.device(kwargs["device"])
        return out
