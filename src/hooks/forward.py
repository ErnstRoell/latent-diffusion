import torch
import structlog

logger = structlog.get_logger()


def forward_hook(module, input, output):
    # log = logger.bind(module=module.__class__.__name__)
    if isinstance(output, torch.Tensor):
        outs = output.shape
    elif isinstance(output, tuple):
        outs = [el.shape for el in output]
    logger.debug(
        f"In shape {module.__class__.__name__}",
        shape=input[0].shape,
    )
    logger.debug(f"Out shape {module.__class__.__name__}", shape=outs)
    logger.debug("---")
