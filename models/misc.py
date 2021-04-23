def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """
    Return the lr of the first parameter group
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
