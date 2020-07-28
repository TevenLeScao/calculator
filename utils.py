import torch


def operations(n_params, batch_size, training_steps):
    # given in PF/s (hence the / 24 / 3600)
    return 6 * n_params * batch_size * training_steps


def excluded_from_params(parameter: torch.nn.Parameter, vocab_size=-1):
    return vocab_size in parameter.shape


def non_emb_param_count(model: torch.nn.Module, vocab_size=-1):
    return sum(p.numel() for p in model.parameters() if not excluded_from_params(p, vocab_size))
