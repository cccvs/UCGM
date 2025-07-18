import torch

class DDIM:

    def __init__(self, **kwargs):
        self.beta_scheduler = kwargs.get("beta_scheduler", "scaled_linear")
        self.num_train_timesteps = torch.tensor(kwargs.get("num_train_timesteps", 1000))
        self.beta_start = torch.tensor(kwargs.get("beta_start", 0.00085))
        self.beta_end = torch.tensor(kwargs.get("beta_end", 0.012))

    def alpha_bar(self, t):
        """
        Compute the continuous version of alpha_bar(t), where t ∈ [0, 1]
        Corresponds to the continuous form of the cumulative product ∏(1-β_i)
        """
        if self.beta_scheduler == "linear":
            # \beta_t = beta_start + t * (beta_end - beta_start)
            k = self.num_train_timesteps * t
            delta_inv = self.num_train_timesteps / (self.beta_end - self.beta_start)
            return torch.exp(
                -k * torch.log(delta_inv) +
                (torch.lgamma((1 - self.beta_start) * delta_inv) - torch.lgamma((1 - self.beta_start) * delta_inv - k))
            )
        elif self.beta_scheduler == "scaled_linear":
            # \beta_t = (self.beta_start ** 0.5 + t * (self.beta_end ** 0.5 - self.beta_start ** 0.5)) ** 2
            k = self.num_train_timesteps * t
            delta_inv = self.num_train_timesteps / (self.beta_end ** 0.5 - self.beta_start ** 0.5)
            x_plus, x_minus = (1 + self.beta_start ** 0.5) * delta_inv, (1 - self.beta_start ** 0.5) * delta_inv
            return torch.exp(
                -2 * k * torch.log(delta_inv) +
                (torch.lgamma(x_minus) - torch.lgamma(x_minus - k)) +
                (torch.lgamma(x_plus + k + 1) - torch.lgamma(x_plus + 1))
            )
        else:
            raise ValueError(f"Unknown beta scheduler: {self.beta_scheduler}")

    def alpha_in(self, t):
        return torch.clamp(1 - self.alpha_bar(t), 0, 1)**0.5

    def gamma_in(self, t):
        return torch.clamp(self.alpha_bar(t), 0, 1)**0.5

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return 0
