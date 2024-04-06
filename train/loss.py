import torch


class LossConstructor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.implem_losses = {
            "mse": MSE,
            "jacobian_reg": JacobianReg_l2,
            "l1_loss": MAE,
        }
        self.config = config
        self.init_losses()
        self.weights = [loss.weight for loss, _ in self.loss]

    def init_losses(self):
        self.loss = [
            (self.implem_losses[loss_name](**self.config[loss_name]), loss_name)
            for loss_name in self.config
            if self.config[loss_name]["active"]
        ]

    def forward(self, pred, target, **kwargs):
        losses = [loss(pred, target, **kwargs) for loss, _ in self.loss]
        stats = dict(
            zip(
                [loss_name for loss_name in self.config],
                [loss.logs for loss, _ in self.loss],
            )
        )
        return (
            sum(losses),
            zip(
                [loss_name for _, loss_name in self.loss],
                [
                    val.detach().item() / self.weights[i]
                    for i, val in enumerate(losses)
                ],
            ),
            stats,
        )


class MSE(torch.nn.Module):
    """
    Mean Squared Error (MSE) Loss
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.weight = kwargs["weight"]
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.logs = None

    def forward(self, pred, target, **kwargs):
        return self.weight * self.criterion(pred, target)


class MAE(torch.nn.Module):
    """
    Mean Absolute Error (MAE) Loss
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.weight = kwargs["weight"]
        self.criterion = torch.nn.L1Loss(reduction="mean")
        self.logs = None

    def forward(self, pred, target, **kwargs):
        return self.weight * self.criterion(pred, target)


class JacobianReg_l2(torch.nn.Module):
    """
    Loss criterion that computes the l2 norm of the Jacobian.
    Arguments:
        max_iter (int, optional): number of iteration in the power method;
        tol (float, optional): convergence criterion of the power method;
        verbose (bool, optional): printing or not the info;
        eval_mode (bool, optional): whether we want to keep the gradients for backprop;
                                    Should be `False' during training.
    Returns:
        z.view(-1) (torch.Tensor of shape [b]): the squared spectral norm of J.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.max_iter = kwargs["max_iter"]
        self.tol = kwargs["tol"]
        self.verbose = kwargs["verbose"]
        self.epsilon = kwargs["epsilon"]
        self.weight = kwargs["weight"]
        self.tensor = (
            torch.cuda.FloatTensor
            if torch.cuda.is_available()
            else torch.FloatTensor
        )

    def reg_fun(self, x, y, eval_mode):
        """ "
        Computes ||dy/dx (x)||_2^2 via a power iteration.
        """
        u = torch.randn_like(x)
        u = u / torch.matmul(
            u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1)
        ).view(u.shape[0], 1, 1, 1)

        z_old = torch.zeros(u.shape[0])

        for it in range(self.max_iter):

            w = torch.ones_like(
                y, requires_grad=True
            )  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(
                torch.autograd.grad(y, x, w, create_graph=True),
                w,
                u,
                create_graph=not eval_mode,
            )[
                0
            ]  # Ju

            (v,) = torch.autograd.grad(
                y, x, v, retain_graph=True, create_graph=True
            )  # vtJt

            z = torch.matmul(
                u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)
            ) / torch.matmul(
                u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1)
            )

            if it > 0:
                rel_var = torch.norm(z - z_old)
                if rel_var < self.tol:
                    if self.verbose:
                        print(
                            "Power iteration converged at iteration: ",
                            it,
                            ", val: ",
                            z,
                        )
                    break
            z_old = z.clone()

            u = v / torch.matmul(
                v.reshape(v.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)
            ).view(v.shape[0], 1, 1, 1)

            if eval_mode:
                w.detach_()
                v.detach_()
                u.detach_()
        return z.view(-1)

    def forward(self, out, x_clean, **kwargs):
        # Convex combination
        tau = torch.rand(x_clean.shape[0], 1, 1, 1).type(self.tensor)
        x_tilde = tau * out.detach() + (1 - tau) * x_clean.detach()
        x_tilde.requires_grad_()

        torch.set_grad_enabled(True)
        out_net = kwargs["model"](x_tilde)
        Q = 2.0 * out_net - x_tilde

        reg_loss = self.reg_fun(x_tilde, Q, kwargs["eval"])
        self.logs = {
            "jacobian_norm_max": reg_loss.max().detach(),
            "jacobian_norm_mean": reg_loss.mean().detach(),
        }
        reg_loss = torch.maximum(
            reg_loss, torch.ones_like(reg_loss) - self.epsilon
        )
        reg_loss = reg_loss.mean()
        return self.weight * reg_loss
