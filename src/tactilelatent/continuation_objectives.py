"""Source gradient gates and a frozen-parent mean anchor; inference is unchanged."""
import math

import torch

from .relational import HELDOUT


class SourceStops:
    """Permanent task/KL gradient gates after consecutive validation deterioration.

    A plateau alone does not stop a source. Its supervision loss must exceed
    its running minimum by the configured relative tolerance for `patience`
    consecutive validations. Histories through the parent epoch initialize
    the state; later historical results are deliberately ignored.
    """
    def __init__(self, sources, config, history):
        if HELDOUT in sources:
            raise ValueError('Held-out target cannot control source gradient gates')
        self.config = config
        self.rows = {s: dict(best=None, bad_epochs=0, stopped_epoch=None) for s in sources}
        for record in history:
            self.update(record['validation'], record['epoch'])

    def update(self, validation, epoch):
        if set(validation) != set(self.rows):
            raise ValueError('Validation sources differ from gradient-gate registry')
        changes = []
        for source, row in self.rows.items():
            value = float(validation[source]['supervision_loss'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid source validation loss')
            row['best'] = value if row['best'] is None else min(value, row['best'])
            worse = value > row['best'] * (1 + self.config['relative_tolerance'])
            row['bad_epochs'] = row['bad_epochs'] + 1 if worse else 0
            if (self.config['enabled'] and row['stopped_epoch'] is None
                    and row['bad_epochs'] >= self.config['patience']):
                row['stopped_epoch'] = epoch
                changes.append(source)
        return changes

    def stopped(self, source):
        if source not in self.rows:
            raise ValueError('Unknown gradient-gate source')
        return self.rows[source]['stopped_epoch'] is not None


def forward(model, encoded, mask, source, stop_trunk=False, anchor_active=False):
    """Keep adapter gradients while detaching a stopped source's task and KL.

    In the combined arm, the parent anchor may still update the trunk on a
    stopped source. It receives the undetached mean through `retention_mu`.
    """
    if source == HELDOUT:
        raise ValueError('MotionTrans cannot enter representation training')
    with torch.set_grad_enabled(torch.is_grad_enabled() and (not stop_trunk or anchor_active)):
        mu, logvar = model.visual.encode(encoded, mask)
    supervised_mu = mu.detach() if stop_trunk else mu
    supervised_logvar = logvar.detach() if stop_trunk else logvar
    z = supervised_mu + torch.randn_like(supervised_mu) * (.5 * supervised_logvar).exp() if model.stochastic and model.training else supervised_mu
    output = dict(mu=supervised_mu, logvar=supervised_logvar, retention_mu=mu)
    if source in model.adapters:
        output['sensor_code'] = model.adapters[source](z)
    else:
        output['pred'] = model.numeric_heads[source](z).reshape(*z.shape[:-1], *model.numeric_shapes[source])
    return output


def retention_loss(mu, target, batch, variance):
    """Mean squared drift per coordinate, scaled by parent train latent variance."""
    if not math.isfinite(variance) or variance <= 0 or mu.shape != target.shape:
        raise ValueError('Invalid parent anchor scale or shapes')
    valid = batch['code_valid'].bool() if 'code_valid' in batch else batch['target_mask'].bool().flatten(2).any(-1)
    if valid.shape != mu.shape[:-1] or not torch.isfinite(target).all():
        raise ValueError('Invalid anchor targets or masks')
    error = (mu.float() - target.detach().float()).square().mean(-1)
    return error[valid].mean() / variance if valid.any() else mu.sum() * 0


def objective(model, item, cfg):
    output = forward(model, item['features'], item['mask'], item['source'],
                     item['stop_trunk'], item['anchor'] is not None)
    from .train_alignment import alignment_loss
    loss, task, kl = alignment_loss(output, item['batch'], item['beta'])
    anchor = output['retention_mu'].sum() * 0
    if item['anchor'] is not None:
        anchor = retention_loss(output['retention_mu'], item['anchor'], item['batch'], cfg['retention']['parent_variance'])
        weight = cfg['retention']['weight'] * min(1., (item['continuation_epoch'] + 1) / cfg['retention']['ramp_epochs'])
        loss = loss + weight * anchor
    return output, loss, task, kl, anchor
