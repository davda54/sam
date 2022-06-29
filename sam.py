import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.0, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert alpha >= 0.0, f"Invalid alpha, should be non-negative: {alpha}"

        defaults = dict(rho=rho, alpha=alpha, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.minimize_surrogate_gap = any(group["alpha"] > 0.0 for group in self.param_groups)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue

                self.state[p]["old_p"] = p.data.clone()
                if self.minimize_surrogate_gap:
                    self.state[p]["old_g"] = p.grad.data.clone()

                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        if self.minimize_surrogate_gap:
            self._gradient_decompose()

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _gradient_decompose(self):
        coeff_nomin, coeff_denom = 0.0, 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                coeff_nomin += (self.state[p]['old_g'] * p.grad).sum()
                coeff_denom += p.grad.pow(2).sum()

        coeff = coeff_nomin / (coeff_denom + 1e-12)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue

                rejection = self.state[p]['old_g'] - coeff * p.grad
                p.grad.data.add_(rejection, alpha=-group["alpha"])

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
