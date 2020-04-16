import torch
from torch.optim import Optimizer

def compute_grad_norm(param_groups):
    grad_norm = 0.
    for group in param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad_norm += torch.sum(torch.mul(p.grad, p.grad))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm

class SPS(Optimizer):
    def __init__(self, params,
                 lr=None,
                 betas=None,
                 weight_decay=None,
                 n_batches_per_epoch=300,
                 init_step_size=1.,
                 c=0.2,
                 gamma=2.0,
                 eta_max=1.,
                 adapt_flag="smooth_iter",
                 eps=1e-6):
        self.eps = eps
        self.params = params
        self.c = c
        self.eta_max = eta_max
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.adapt_flag = adapt_flag
        self.step_size_max = 0.
        self.n_batches_per_epoch = n_batches_per_epoch
        super(SPS, self).__init__(params, {})

    def step(self, closure=None, loss=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            loss (scalar): The current loss
        """

        # we need the loss
        assert loss is not None

        # calculate gradient norm of all parameters (-> scalar) 
        grad_norm = compute_grad_norm(self.param_groups)

        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('Does not support sparse gradients, consider SparseAdam instad.')

                # grab state for current parameter
                state = self.state[p]

                # init state
                if len(state) == 0:
                    state['step'] = 0
                    state['step_size_avg'] = 0.
                    state['n_forwards'] = 0
                    state['n_backwards'] = 0
                    state['step_size'] = self.init_step_size

                # increment steps
                state['step'] += 1

                # adaptation
                step_size = loss / (self.c * (grad_norm)**2 + self.eps)

                # adjust step size: based on an upper bound
                if self.adapt_flag in ['constant']:
                    if loss < 0.:
                        step_size = 0.
                    else:
                        if self.eta_max is None:
                            step_size = step_size.item()
                        else:
                            a, b = self.eta_max, step_size.item()
                            step_size = min(a, b)

                # adjust step size: smooth
                elif self.adapt_flag in ['smooth_iter']:
                    coeff = self.gamma ** (1. / self.n_batches_per_epoch)
                    a, b = coeff * state['step_size'], step_size.item()
                    step_size =  min(a, b)

                else:
                    raise Exception("adapt_flag not supported.")

                # perform SGD update with step size
                p.data.add_(-step_size * grad)
                
                # metrics
                if state['step'] % int(self.n_batches_per_epoch) == 1:
                    # reset step size avg for each new epoch
                    state['step_size_avg'] = 0.
                state['step_size_avg'] += (step_size / self.n_batches_per_epoch)
                state['n_forwards'] += 1
                state['n_backwards'] += 1
                state['step_size'] = step_size
                state['grad_norm'] = grad_norm.item()

        return loss
