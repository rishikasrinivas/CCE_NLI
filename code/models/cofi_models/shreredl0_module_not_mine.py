import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from transformers.utils import logging

logger = logging.get_logger(__name__)

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
MEAN = {
    "hidden": 5.0,           # was 10 — most important change
    "head": 5.0,
    "head_layer": 5.0,
    "mlp": 5.0,
    "intermediate": 0,       # 0 → droprate mean (~0.0), OK for intermediate
    "final_mlp_hidden": 5.0,
}
class Mask(nn.Module):
    def __init__(
      self,
      name: str,
      mask_shape: List[int],
      num_params_per_mask: int,
      mask_output_shape: List[int],
      device: str,
      target_mask_size: Optional[int] = None,
      eval_target_model: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.num_params_per_mask = num_params_per_mask
        self.mask_output_shape = mask_output_shape
        self.target_mask_size = target_mask_size
        self.eval_target_model = eval_target_model

        self.droprate_init = 0.5
        self.temperature = 2.0 / 3.0
        self.magical_number = 0.8
        self.device = device

        self.z_loga = self._initialize_mask(mask_shape)
        self.mask_size = self.z_loga.shape[-1]

    def get_size(self):
        return self.mask_output_shape

    def _param_init_fn(self, tensor: nn.Parameter) -> None:
        # Sheared-style: mean=5 for all masks (including hidden)
        #mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        mean = MEAN[self.name]
        if mean == 0:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)

    def _initialize_mask(self, mask_shape: List[int]) -> nn.Parameter:
        z_loga = nn.Parameter(torch.empty(*mask_shape, device=self.device))
        self._param_init_fn(z_loga)
        return z_loga

    def cdf_qz(self, z_loga: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z_loga is None:
            z_loga = self.z_loga
        xn = (0 - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - z_loga).clamp(
            min=epsilon, max=1 - epsilon
        )

    def get_eps(self, size: torch.Size) -> torch.Tensor:
        eps = torch.empty(size, device=self.z_loga.device).uniform_(epsilon, 1 - epsilon)
        return Variable(eps)

    def quantile_concrete(self, eps: torch.Tensor) -> torch.Tensor:
        y = torch.sigmoid(
            (torch.log(eps) - torch.log(1 - eps) + self.z_loga) / self.temperature
        )
        return y * (limit_b - limit_a) + limit_a

    def sample_z(self) -> torch.Tensor:
        eps = self.get_eps(torch.Size(self.z_loga.shape))
        z = self.quantile_concrete(eps)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z.reshape(*self.mask_output_shape)

    def _deterministic_z(self, z_loga: torch.Tensor) -> torch.Tensor:
        if self.target_mask_size is not None and self.eval_target_model:
            expected_num_zeros = self.mask_size - self.target_mask_size
        else:
            expected_score = 1 - self.cdf_qz(z_loga)
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = z_loga.numel() - expected_num_nonzeros.item()

        try:
            num_zeros = round(expected_num_zeros)
        except Exception:
            logger.error("num_zeros is NaN for mask %s", self.name)
            sys.exit(1)

        soft_mask = torch.sigmoid(z_loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0.0, device=z_loga.device)
            else:
                k = min(num_zeros, soft_mask.numel())
                _, indices = torch.topk(soft_mask, k=k, largest=False)
                soft_mask = soft_mask.clone()
                soft_mask[indices] = 0.0
        return soft_mask

    def deterministic_z(self) -> torch.Tensor:
        if self.z_loga.ndim == 1:
            z = self._deterministic_z(self.z_loga)
            return z.reshape(*self.mask_output_shape)

        z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
        rows = [self._deterministic_z(z_loga[i]) for i in range(z_loga.shape[0])]
        return torch.stack(rows).reshape(*self.mask_output_shape)

    def constrain_parameters(self) -> None:
        self.z_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def calculate_expected_score_sparsity(self) -> torch.Tensor:
        return 1 - self.cdf_qz()


class L0Module_LLAMA(nn.Module):
    """
    CoFi-style L0 for GQA LLaMA + NLI classifier head.

    pruning_type tokens (split on '+'):
    hidden, structured_heads, structured_mlp, layer, final_mlp_hidden
    """

    PRUNING_ALIASES = {
      "structured_heads": "head",
      "structured_mlp": "intermediate",
    }

    def __init__(
      self,
      config=None,
      model_name=None,
      droprate_init: float = 0.5,
      temperature: float = 1.0 / 3.0,
      lagrangian_warmup: int = 0,
      start_sparsity: float = 0.0,
      target_sparsity: float = 0.0,
      args=None,
      full_model_size=None,
      pruning_type: str = "structured_heads+structured_mlp+hidden+layer+final_mlp_hidden",
      magical_number: float = 0.8,
      device: str = "cuda",
    ):
        super().__init__()

        self.config = config
        self.model_name = model_name
        self.final_mlp_hidden = 1024
        self.out_params = 3

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.dim_per_head = self.hidden_size // self.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.kv_dim = self.num_key_value_heads * self.dim_per_head

        # Per-layer parameter budgets (GQA)
        self.params_per_head_layer = (
            self.hidden_size * self.hidden_size  # q_proj
            + self.hidden_size * self.kv_dim  # k_proj
            + self.hidden_size * self.kv_dim  # v_proj
            + self.hidden_size * self.hidden_size  # o_proj
        )
        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 3
        self.params_per_head = self.params_per_head_layer // self.num_key_value_heads
        self.params_per_intermediate_dim = (
            self.params_per_mlp_layer // self.intermediate_size
        )
        self.params_finalmlp_layer = (
            self.hidden_size * 4 * self.final_mlp_hidden
            + self.final_mlp_hidden * self.out_params
        )

        self.attn_cost_per_hidden_kv_pair = (
            2 * self.num_key_value_groups * self.dim_per_head  # Q + O
            + 2 * self.dim_per_head  # K + V
        )

        self.start_sparsity = start_sparsity
        self.lagrangian_warmup_steps = lagrangian_warmup
        self.target_sparsity = target_sparsity
        self.device = device
        self.magical_number = magical_number

        raw_modules = pruning_type.split("+")
        self.pruning_modules = [self.PRUNING_ALIASES.get(m, m) for m in raw_modules]

        self.lambda_1 = nn.Parameter(torch.tensor(0.0, device=device))
        self.lambda_2 = nn.Parameter(torch.tensor(0.0, device=device))

        self.masks = nn.ModuleDict()
        for module_name in raw_modules:
            self._initialize_one_module(module_name)

        self.prunable_model_size = self.calculate_prunable_model_size()

        logger.info("********** Initializing L0 Module **********")
        for name, mask in self.masks.items():
            logger.info("***** %s *****", name)
            logger.info("z.shape %s", mask.z_loga.shape)
            logger.info("mask_size %s", mask.mask_size)
        logger.info("prunable model size: %s", self.prunable_model_size)

    def set_lagrangian_warmup_steps(self, lagrangian_warmup: int) -> None:
        self.lagrangian_warmup_steps = lagrangian_warmup

    def _backbone_size(self) -> float:
        L = self.num_hidden_layers
        return (self.params_per_head_layer + self.params_per_mlp_layer) * L

    def calculate_prunable_model_size(self) -> float:
        """
        Denominator for expected_sparsity. Must match get_expected_num_params at all-ones scores.

        Sheared rule: if "hidden" is pruned, the budget is the full backbone (+ NLI head),
        because hidden masks feature indices shared by attn and MLP (and 4×hidden classifier input).
        """
        L = self.num_hidden_layers
        head_budget = self.params_per_head_layer * L
        mlp_budget = self.params_per_mlp_layer * L

        if "hidden" in self.pruning_modules:
            return self._backbone_size() + self.params_finalmlp_layer

        size = 0.0
        if "head" in self.pruning_modules or "head_layer" in self.pruning_modules:
            size += head_budget
        if "intermediate" in self.pruning_modules or "mlp" in self.pruning_modules:
            size += mlp_budget
        if "final_mlp_hidden" in self.pruning_modules:
            size += self.params_finalmlp_layer
        return size

    def _initialize_one_module(self, module_name: str) -> None:
        if module_name in ("structured_mlp", "intermediate"):
            self._initialize_structured_mlp()
        elif module_name in ("structured_heads", "head"):
            self._initialize_structured_head()
        elif module_name == "hidden":
            self._initialize_hidden()
        elif module_name == "layer":
            self._initialize_whole_mlp()
            self._initialize_layer_structured_heads()
        elif module_name == "final_mlp_hidden":
            self._initialize_final_hidden_layer_mlp()
        else:
            raise ValueError(f"Unknown pruning module: {module_name}")

    def _initialize_hidden(self) -> None:
        num_params_per_mask = (
            self.hidden_size  # q in
            + self.kv_dim  # k in
            + self.kv_dim  # v in
            + self.hidden_size  # o out
            + 3 * self.intermediate_size  # gate, up, down
        )
        self.masks["hidden"] = Mask(
            name="hidden",
            mask_shape=[self.hidden_size],
            num_params_per_mask=num_params_per_mask,
            mask_output_shape=[self.hidden_size],
            device=self.device,
        )

    def _initialize_structured_head(self) -> None:
        self.masks["head"] = Mask(
            name="head",
            mask_shape=[self.num_hidden_layers, self.num_key_value_heads],
            num_params_per_mask=self.params_per_head,
            mask_output_shape=[self.num_hidden_layers, 1, self.num_key_value_heads, 1],
            device=self.device,
        )

    def _initialize_final_hidden_layer_mlp(self) -> None:
        self.masks["final_mlp_hidden"] = Mask(
            name="final_mlp_hidden",
            mask_shape=[self.final_mlp_hidden],
            num_params_per_mask=self.out_params,
            mask_output_shape=[self.final_mlp_hidden],
            device=self.device,
        )

    def _initialize_layer_structured_heads(self) -> None:
        self.masks["head_layer"] = Mask(
            name="head_layer",
            mask_shape=[self.num_hidden_layers],
            num_params_per_mask=self.params_per_head * self.num_key_value_heads,
            mask_output_shape=[self.num_hidden_layers],
            device=self.device,
        )

    def _initialize_structured_mlp(self) -> None:
        self.masks["intermediate"] = Mask(
            name="intermediate",
            mask_shape=[self.num_hidden_layers, self.intermediate_size],
            num_params_per_mask=self.params_per_intermediate_dim,
            mask_output_shape=[self.num_hidden_layers, 1, 1, self.intermediate_size],
            device=self.device,
        )

    def _initialize_whole_mlp(self) -> None:
        self.masks["mlp"] = Mask(
            name="mlp",
            mask_shape=[self.num_hidden_layers],
            num_params_per_mask=self.params_per_mlp_layer,
            mask_output_shape=[self.num_hidden_layers],
            device=self.device,
        )

    def constrain_parameters(self) -> None:
        for mask in self.masks.values():
            mask.constrain_parameters()

    def calculate_expected_score_sparsity(self) -> Dict[str, torch.Tensor]:
        return {k: m.calculate_expected_score_sparsity() for k, m in self.masks.items()}

    def _ones_head_score(self, device: torch.device) -> torch.Tensor:
        return torch.ones(
            self.num_hidden_layers, self.num_key_value_heads, device=device
        )

    def _ones_int_score(self, device: torch.device) -> torch.Tensor:
        return torch.ones(
            self.num_hidden_layers, self.intermediate_size, device=device
        )

    def transform_scores_for_head(
      self, expected_scores: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        device = next(iter(expected_scores.values())).device
        head_score = expected_scores.get("head", self._ones_head_score(device))

        head_layer_score = None
        if "head_layer" in expected_scores:
            head_layer_score = expected_scores["head_layer"].view(-1, 1)
        elif "layer" in expected_scores:
            head_layer_score = expected_scores["layer"].view(-1, 1)

        return head_layer_score, head_score

    def transform_scores_for_mlp(
      self, expected_scores: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        device = next(iter(expected_scores.values())).device
        intermediate_score = expected_scores.get("intermediate", self._ones_int_score(device))

        mlp_score = None
        if "mlp" in expected_scores:
            mlp_score = expected_scores["mlp"].unsqueeze(-1)
        elif "layer" in expected_scores:
            mlp_score = expected_scores["layer"].unsqueeze(-1)

        return mlp_score, intermediate_score

    def get_expected_num_params(self, expected_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(iter(expected_scores.values())).device
        num_parameters = torch.tensor(0.0, device=device)

        head_layer_score, head_score = self.transform_scores_for_head(expected_scores)
        mlp_score, int_score = self.transform_scores_for_mlp(expected_scores)

        if head_layer_score is not None:
            head_score = head_layer_score * head_score
        if mlp_score is not None:
            int_score = mlp_score * int_score

        if "hidden" in expected_scores:
            hidden_score = expected_scores["hidden"]

            active_hidden = hidden_score.sum()
            active_heads = head_score.sum()
            active_int = int_score.sum()

            num_parameters = num_parameters + active_hidden * active_heads * self.attn_cost_per_hidden_kv_pair
            num_parameters = num_parameters + active_hidden * active_int * 3

            # NLI head: 4×hidden input (tied to hidden mask even without final_mlp_hidden mask)
            if "final_mlp_hidden" in expected_scores:
                final_hidden_score = expected_scores["final_mlp_hidden"]
            else:
                final_hidden_score = torch.ones(self.final_mlp_hidden, device=device)

            active_final_hidden = final_hidden_score.sum()
            active_final_input = active_hidden * 4
            num_parameters = num_parameters + active_final_input * active_final_hidden
            num_parameters = num_parameters + active_final_hidden * self.out_params
        else:
            num_parameters = num_parameters + head_score.sum() * self.params_per_head
            num_parameters = num_parameters + int_score.sum() * self.params_per_intermediate_dim
            if "final_mlp_hidden" in expected_scores:
                num_parameters = num_parameters + expected_scores["final_mlp_hidden"].sum() * self.out_params

        return num_parameters

    def get_target_sparsity(
      self, pruned_steps: int, full_sparsity: Optional[float] = None
    ) -> float:
        target = self.target_sparsity if full_sparsity is None else full_sparsity
        if getattr(self, "lagrangian_warmup_steps", 0) > 0:
            t = min(1.0, pruned_steps / self.lagrangian_warmup_steps)
            target = (target - self.start_sparsity) * t + self.start_sparsity
        return target

    def lagrangian_regularization(
      self, pruned_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        target_sparsity = self.get_target_sparsity(pruned_steps, self.target_sparsity)
        expected_scores = self.calculate_expected_score_sparsity()
        expected_size = self.get_expected_num_params(expected_scores)

        if self.prunable_model_size <= 0:
            zero = torch.tensor(0.0, device=expected_size.device, requires_grad=True)
            return zero, zero, target_sparsity

        expected_sparsity = 1 - expected_size / self.prunable_model_size
        lagrangian_loss = self.lambda_1 * (expected_sparsity - target_sparsity) + self.lambda_2 * (
            expected_sparsity - target_sparsity
        ) ** 2
        return lagrangian_loss, expected_sparsity, target_sparsity

    def get_z_from_zs(self, zs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        out = {}
        for name, mask in self.masks.items():
            z = zs.get(f"{name}_z", np.ones(mask.get_size()))
            if torch.is_tensor(z):
                z = z.squeeze().detach().cpu().numpy() > 0
            out[name] = z
        return out

    def calculate_model_size(self, zs: Dict[str, torch.Tensor]) -> Dict:
        nz = self.get_z_from_zs(zs)

        hidden_z = nz.get("hidden", np.ones(self.hidden_size))
        head_z = nz.get("head", np.ones((self.num_hidden_layers, self.num_key_value_heads)))
        intermediate_z = nz.get(
            "intermediate", np.ones((self.num_hidden_layers, self.intermediate_size))
        )
        head_layer_z = nz.get("head_layer", np.ones(self.num_hidden_layers)).reshape(-1, 1)
        mlp_z = nz.get("mlp", np.ones(self.num_hidden_layers)).reshape(-1, 1)

        head_mask = head_z.reshape(self.num_hidden_layers, self.num_key_value_heads) * head_layer_z
        intermediate_mask = (
            intermediate_z.reshape(self.num_hidden_layers, self.intermediate_size) * mlp_z
        )

        head_nums = np.outer(head_mask.reshape(-1), hidden_z).sum()
        intermediate_nums = np.outer(intermediate_mask.reshape(-1), hidden_z).sum()

        remaining_model_size = (
            head_nums * self.attn_cost_per_hidden_kv_pair + intermediate_nums * 3
        )

        mlp_final_hidden = nz.get("final_mlp_hidden", np.ones(self.final_mlp_hidden))
        mlp_final_input = np.concatenate([hidden_z, hidden_z, hidden_z, hidden_z])
        final = mlp_final_input.sum() * mlp_final_hidden.sum() + mlp_final_hidden.sum() * self.out_params
        remaining_model_size += final

        pruned_model_size = self.prunable_model_size - remaining_model_size

        results = {
            "hidden_dims": int(hidden_z.sum()),
            "intermediate_dims": intermediate_mask.sum(-1).astype(int).tolist(),
            "head_nums": head_mask.sum(-1).astype(int).tolist(),
            "final_mlp_hidden": int(mlp_final_hidden.sum()),
            "pruned_params": pruned_model_size,
            "remaining_params": remaining_model_size,
            "pruned_model_sparsity": pruned_model_size / max(self.prunable_model_size, 1),
        }
        if "head_layer" in self.masks:
            results["head_layer"] = head_layer_z.reshape(-1).astype(int).tolist()
        if "mlp" in self.masks:
            results["mlp"] = mlp_z.reshape(-1).astype(int).tolist()

        return results

    def forward(self, training: bool = True) -> Dict[str, torch.Tensor]:
        zs = {}
        if training:
            for name, mask in self.masks.items():
                zs[f"{name}_z"] = mask.sample_z()
        else:
            with torch.no_grad():
                for name, mask in self.masks.items():
                    zs[f"{name}_z"] = mask.deterministic_z()

        # Trainer: when hidden is pruned, mask the 4×hidden NLI input
        if "hidden_z" in zs and "final_mlp_inp_z" not in zs:
            h = zs["hidden_z"].reshape(-1)
            zs["final_mlp_inp_z"] = torch.cat([h, h, h, h], dim=0)

        return zs