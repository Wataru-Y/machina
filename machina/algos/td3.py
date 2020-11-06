import torch
import torch.nn as nn

from machina import loss_functional as lf
from machina import logger


def train(traj,
          pol, targ_pol, qfs, targ_qfs,
          optim_pol, optim_qfs,
          epoch, batch_size,  # optimization hypers
          tau, gamma,  # advantage estimation
          pol_update=True, 
          log_enable=True,
          max_grad_norm=0.5,
          ):

    pol_losses = []
    _qf_losses = []
    if log_enable:
        logger.log("Optimizing...")

    for batch in traj.random_batch(batch_size, epoch):
        qf_losses = lf.td3(qfs, targ_qfs, targ_pol, batch, gamma, continuous=True, deterministic=True, sampling=1)
        
        for qf, optim_qf, qf_loss in zip(qfs, optim_qfs, qf_losses):
            optim_qf.zero_grad()
            qf_loss.backward()
            #torch.nn.utils.clip_grad_norm_(qf.parameters(), max_grad_norm)
            optim_qf.step()
        
        _qf_losses.append((sum(qf_losses) / len(qf_losses)).detach().cpu().numpy())

        if pol_update:
            pol_loss = lf.ag(pol, qfs[0], batch, no_noise=True)
            optim_pol.zero_grad()
            pol_loss.backward()
            #torch.nn.utils.clip_grad_norm_(pol.parameters(), max_grad_norm)
            optim_pol.step()
    
            for p, targ_p in zip(pol.parameters(), targ_pol.parameters()):
                targ_p.detach().copy_((1 - tau) * targ_p.detach() + tau * p.detach())
            
            for qf, targ_qf in zip(qfs, targ_qfs):
                for q, targ_q in zip(qf.parameters(), targ_qf.parameters()):
                    targ_q.detach().copy_((1 - tau) * targ_q.detach() + tau * q.detach())

            pol_losses.append(pol_loss.detach().cpu().numpy())
            
    if log_enable:
                logger.log("Optimization finished!")
    if pol_update:

        return dict(
            PolLoss=pol_losses,
            QfLoss=_qf_losses,
            )
        
    else:

        return dict(
            QfLoss=_qf_losses,
            )
