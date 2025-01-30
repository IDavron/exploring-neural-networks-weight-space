import torch
import torch.nn.functional as F
import numpy as np

def sample_gaussian(centers, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), np.sqrt(var) * torch.eye(2))
    n = centers.shape[0]
    centers = centers * scale
    noise = m.sample((n,)).to(centers.device)
    data = []
    for i in range(n):
        data.append(centers[i] + noise[i])
    data = torch.stack(data)
    return data

def sample_vae(model, angle):
    a = torch.tensor([angle * torch.pi / 180])
    sin = torch.sin(a)
    cos = torch.cos(a)
    center = torch.stack([cos, sin], dim=1)
    z = sample_gaussian(center, scale=3, var=0.1)
    params = model.decoder(z).squeeze()

    return params


def sample_flow(model, angle, num_iter=100, prior_dim=33, prior_sd=1):
    gaussian = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(prior_dim), np.sqrt(prior_sd) * torch.eye(prior_dim))

    sample = gaussian.sample((1,))
    angle = torch.tensor([angle*torch.pi/180])
    for i in np.linspace(0, 1, num_iter, endpoint=False):
        t = torch.tensor([i], dtype=torch.float32)
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        path = model(torch.cat([sample, t[:, None], sin[:, None], cos[:, None]], dim=-1))
        sample += (0.01 * path)
    return sample[0]


def reconstruct_xt(noise, x_t, t, betas):
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        sqrt_inv_alphas_cumprod = torch.sqrt(1 / alphas_cumprod)
        sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / alphas_cumprod - 1)

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        s1 = sqrt_inv_alphas_cumprod[t]
        s2 = sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        pred_x0 = s1 * x_t - s2 * noise

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        s1 = posterior_mean_coef1[t]
        s2 = posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        x_t = s1 * pred_x0 + s2 * x_t

        variance = 0
        if t > 0:
            variance = betas[t] * (1. - alphas_cumprod_prev[t]) / (1. - alphas_cumprod[t])
            variance = variance.clip(1e-20)
            noise = torch.randn_like(noise)
            variance = (variance ** 0.5) * noise

        pred_prev_sample = x_t + variance

        return pred_prev_sample

def sample_diffusion(model, angle, num_timesteps=1000, betas=None, prior_dim=33):
    if(betas is None):
        betas = torch.tensor(np.linspace(1e-4, 0.02, num_timesteps), dtype=torch.float32)
    else:
        betas = betas

    a = torch.tensor([angle * np.pi / 180])
    sin = torch.sin(a)
    cos = torch.cos(a)
    a = torch.cat([sin[None, :], cos[None, :]], dim=1)

    sample = torch.randn(1, prior_dim)
    timesteps = list(range(num_timesteps))[::-1]
    for _, t in enumerate(timesteps):
        t = torch.from_numpy(np.repeat(t, 1)).long()
        with torch.no_grad():
            residual = model(sample, t, a)
        sample = reconstruct_xt(residual, sample, t[0], betas)

    return sample[0]

