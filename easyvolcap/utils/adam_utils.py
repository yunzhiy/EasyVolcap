import math
import torch
from typing import List, Union, Optional
from torch.utils.cpp_extension import load
from torch.optim import Adam
from easyvolcap.utils.console_utils import *

class MyFusedAdam(Adam):
    def step(self, closure=None):
        """Perform a single optimization step.

        Will disrespect weight decay, but significantly reduce kernel launches

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            from easyvolcap.utils.adam_utils import _single_tensor_adam
            _single_tensor_adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=getattr(self, "amsgrad", False),
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=getattr(self, "maximize", False),
                capturable=getattr(self, "capturable", False),
                differentiable=getattr(self, "differentiable", False),
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        # Older version of PyTorch doesn't have this method
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state['step'] = (
                        torch.tensor(0.0, dtype=torch.float32)
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                state_steps.append(state['step'])
        return has_complex



@torch.jit.script
def adam(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
):
    # update step
    step_t += 1

    # Decay the first and second moment running average coefficient
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

    step = step_t.item()

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = math.sqrt(bias_correction2)

    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    param.addcdiv_(exp_avg, denom, value=-step_size)


module = load(
    name='fused_adam',
    sources=[f'{dirname(__file__)}/src/fused_adam.cu'],
    extra_include_paths=[os.environ.get('CUDA_HOME', '/usr/local/cuda') + "/include"],
    extra_cuda_cflags=["--expt-relaxed-constexpr",
                       "-O2"],
    verbose=True
)


def fused_adam(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
):
    step_t += 1
    return module.fused_adam(param, grad, exp_avg, exp_avg_sq, step_t.item(), beta1, beta2, lr, eps)


def _single_tensor_adam(params: List[torch.Tensor],
                        grads: List[torch.Tensor],
                        exp_avgs: List[torch.Tensor],
                        exp_avg_sqs: List[torch.Tensor],
                        max_exp_avg_sqs: List[torch.Tensor],
                        state_steps: List[torch.Tensor],
                        grad_scale: Optional[torch.Tensor],
                        found_inf: Optional[torch.Tensor],
                        *,
                        amsgrad: bool,
                        has_complex: bool,
                        beta1: float,
                        beta2: float,
                        lr: Union[float, torch.Tensor],
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool
                        ):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        fused_adam(param, grad, exp_avg, exp_avg_sq, step_t, beta1, beta2, lr, eps)
