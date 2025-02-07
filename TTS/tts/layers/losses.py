import math

import numpy as np
import torch
from coqpit import Coqpit
from torch import nn
from torch.nn import functional

from TTS.tts.utils.helpers import sequence_mask
from TTS.tts.utils.ssim import ssim
from TTS.utils.audio import TorchSTFT

from TTS.style_encoder.losses import *

# pylint: disable=abstract-method
# relates https://github.com/pytorch/pytorch/issues/42305
class L1LossMasked(nn.Module):
    def __init__(self, seq_len_norm):
        super().__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            x: B x T X D
            target: B x T x D
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.l1_loss(x * mask, target * mask, reduction="none")
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.l1_loss(x * mask, target * mask, reduction="sum")
            loss = loss / mask.sum()
        return loss


class MSELossMasked(nn.Module):
    def __init__(self, seq_len_norm):
        super().__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            - x: :math:`[B, T, D]`
            - target: :math:`[B, T, D]`
            - length: :math:`B`
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.mse_loss(x * mask, target * mask, reduction="none")
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.mse_loss(x * mask, target * mask, reduction="sum")
            loss = loss / mask.sum()
        return loss


class SSIMLoss(torch.nn.Module):
    """SSIM loss as explained here https://en.wikipedia.org/wiki/Structural_similarity"""

    def __init__(self):
        super().__init__()
        self.loss_func = ssim

    def forward(self, y_hat, y, length=None):
        """
        Args:
            y_hat (tensor): model prediction values.
            y (tensor): target values.
            length (tensor): length of each sample in a batch.
        Shapes:
            y_hat: B x T X D
            y: B x T x D
            length: B
         Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        if length is not None:
            m = sequence_mask(sequence_length=length, max_len=y.size(1)).unsqueeze(2).float().to(y_hat.device)
            y_hat, y = y_hat * m, y * m
        return 1 - self.loss_func(y_hat.unsqueeze(1), y.unsqueeze(1))


class AttentionEntropyLoss(nn.Module):
    # pylint: disable=R0201
    def forward(self, align):
        """
        Forces attention to be more decisive by penalizing
        soft attention weights

        TODO: arguments
        TODO: unit_test
        """
        entropy = torch.distributions.Categorical(probs=align).entropy()
        loss = (entropy / np.log(align.shape[1])).mean()
        return loss


class BCELossMasked(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Shapes:
            x: B x T
            target: B x T
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        if length is not None:
            mask = sequence_mask(sequence_length=length, max_len=target.size(1)).float()
            x = x * mask
            target = target * mask
            num_items = mask.sum()
        else:
            num_items = torch.numel(x)
        loss = functional.binary_cross_entropy_with_logits(x, target, pos_weight=self.pos_weight, reduction="sum")
        loss = loss / num_items
        return loss


class DifferentailSpectralLoss(nn.Module):
    """Differential Spectral Loss
    https://arxiv.org/ftp/arxiv/papers/1909/1909.10302.pdf"""

    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, x, target, length=None):
        """
         Shapes:
            x: B x T
            target: B x T
            length: B
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        x_diff = x[:, 1:] - x[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        if length is None:
            return self.loss_func(x_diff, target_diff)
        return self.loss_func(x_diff, target_diff, length - 1)


class GuidedAttentionLoss(torch.nn.Module):
    def __init__(self, sigma=0.4):
        super().__init__()
        self.sigma = sigma

    def _make_ga_masks(self, ilens, olens):
        B = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        ga_masks = torch.zeros((B, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            ga_masks[idx, :olen, :ilen] = self._make_ga_mask(ilen, olen, self.sigma)
        return ga_masks

    def forward(self, att_ws, ilens, olens):
        ga_masks = self._make_ga_masks(ilens, olens).to(att_ws.device)
        seq_masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = ga_masks * att_ws
        loss = torch.mean(losses.masked_select(seq_masks))
        return loss

    @staticmethod
    def _make_ga_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen).to(olen), torch.arange(ilen).to(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = sequence_mask(ilens)
        out_masks = sequence_mask(olens)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


class Huber(nn.Module):
    # pylint: disable=R0201
    def forward(self, x, y, length=None):
        """
        Shapes:
            x: B x T
            y: B x T
            length: B
        """
        mask = sequence_mask(sequence_length=length, max_len=y.size(1)).unsqueeze(2).float()
        return torch.nn.functional.smooth_l1_loss(x * mask, y * mask, reduction="sum") / mask.sum()


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = torch.nn.functional.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss = total_loss + loss

        total_loss = total_loss / attn_logprob.shape[0]
        return total_loss


########################
# MODEL LOSS LAYERS
########################


class TacotronLoss(torch.nn.Module):
    """Collection of Tacotron set-up based on provided config."""

    def __init__(self, c, ga_sigma=0.4):
        super().__init__()
        self.stopnet_pos_weight = c.stopnet_pos_weight
        self.ga_alpha = c.ga_alpha
        self.decoder_diff_spec_alpha = c.decoder_diff_spec_alpha
        self.postnet_diff_spec_alpha = c.postnet_diff_spec_alpha
        self.decoder_alpha = c.decoder_loss_alpha
        self.postnet_alpha = c.postnet_loss_alpha
        self.decoder_ssim_alpha = c.decoder_ssim_alpha
        self.postnet_ssim_alpha = c.postnet_ssim_alpha
        self.config = c

        # postnet and decoder loss
        if c.loss_masking:
            self.criterion = L1LossMasked(c.seq_len_norm) if c.model in ["Tacotron"] else MSELossMasked(c.seq_len_norm)
        else:
            self.criterion = nn.L1Loss() if c.model in ["Tacotron"] else nn.MSELoss()
        # guided attention loss
        if c.ga_alpha > 0:
            self.criterion_ga = GuidedAttentionLoss(sigma=ga_sigma)
        # differential spectral loss
        if c.postnet_diff_spec_alpha > 0 or c.decoder_diff_spec_alpha > 0:
            self.criterion_diff_spec = DifferentailSpectralLoss(loss_func=self.criterion)
        # ssim loss
        if c.postnet_ssim_alpha > 0 or c.decoder_ssim_alpha > 0:
            self.criterion_ssim = SSIMLoss()
        # stopnet loss
        # pylint: disable=not-callable
        self.criterion_st = BCELossMasked(pos_weight=torch.tensor(self.stopnet_pos_weight)) if c.stopnet else None

    def forward(
        self,
        postnet_output,
        decoder_output,
        mel_input,
        linear_input,
        stopnet_output,
        stopnet_target,
        stop_target_length,
        output_lens,
        decoder_b_output,
        alignments,
        alignment_lens,
        alignments_backwards,
        input_lens,
    ):

        # decoder outputs linear or mel spectrograms for Tacotron and Tacotron2
        # the target should be set acccordingly
        postnet_target = linear_input if self.config.model.lower() in ["tacotron"] else mel_input

        return_dict = {}
        # remove lengths if no masking is applied
        if not self.config.loss_masking:
            output_lens = None
        # decoder and postnet losses
        if self.config.loss_masking:
            if self.decoder_alpha > 0:
                decoder_loss = self.criterion(decoder_output, mel_input, output_lens)
            if self.postnet_alpha > 0:
                postnet_loss = self.criterion(postnet_output, postnet_target, output_lens)
        else:
            if self.decoder_alpha > 0:
                decoder_loss = self.criterion(decoder_output, mel_input)
            if self.postnet_alpha > 0:
                postnet_loss = self.criterion(postnet_output, postnet_target)
        loss = self.decoder_alpha * decoder_loss + self.postnet_alpha * postnet_loss
        return_dict["decoder_loss"] = decoder_loss
        return_dict["postnet_loss"] = postnet_loss

        stop_loss = (
            self.criterion_st(stopnet_output, stopnet_target, stop_target_length)
            if self.config.stopnet
            else torch.zeros(1)
        )
        loss += stop_loss
        return_dict["stopnet_loss"] = stop_loss

        # backward decoder loss (if enabled)
        if self.config.bidirectional_decoder:
            if self.config.loss_masking:
                decoder_b_loss = self.criterion(torch.flip(decoder_b_output, dims=(1,)), mel_input, output_lens)
            else:
                decoder_b_loss = self.criterion(torch.flip(decoder_b_output, dims=(1,)), mel_input)
            decoder_c_loss = torch.nn.functional.l1_loss(torch.flip(decoder_b_output, dims=(1,)), decoder_output)
            loss += self.decoder_alpha * (decoder_b_loss + decoder_c_loss)
            return_dict["decoder_b_loss"] = decoder_b_loss
            return_dict["decoder_c_loss"] = decoder_c_loss

        # double decoder consistency loss (if enabled)
        if self.config.double_decoder_consistency:
            if self.config.loss_masking:
                decoder_b_loss = self.criterion(decoder_b_output, mel_input, output_lens)
            else:
                decoder_b_loss = self.criterion(decoder_b_output, mel_input)
            # decoder_c_loss = torch.nn.functional.l1_loss(decoder_b_output, decoder_output)
            attention_c_loss = torch.nn.functional.l1_loss(alignments, alignments_backwards)
            loss += self.decoder_alpha * (decoder_b_loss + attention_c_loss)
            return_dict["decoder_coarse_loss"] = decoder_b_loss
            return_dict["decoder_ddc_loss"] = attention_c_loss

        # guided attention loss (if enabled)
        if self.config.ga_alpha > 0:
            ga_loss = self.criterion_ga(alignments, input_lens, alignment_lens)
            loss += ga_loss * self.ga_alpha
            return_dict["ga_loss"] = ga_loss

        # decoder differential spectral loss
        if self.config.decoder_diff_spec_alpha > 0:
            decoder_diff_spec_loss = self.criterion_diff_spec(decoder_output, mel_input, output_lens)
            loss += decoder_diff_spec_loss * self.decoder_diff_spec_alpha
            return_dict["decoder_diff_spec_loss"] = decoder_diff_spec_loss

        # postnet differential spectral loss
        if self.config.postnet_diff_spec_alpha > 0:
            postnet_diff_spec_loss = self.criterion_diff_spec(postnet_output, postnet_target, output_lens)
            loss += postnet_diff_spec_loss * self.postnet_diff_spec_alpha
            return_dict["postnet_diff_spec_loss"] = postnet_diff_spec_loss

        # decoder ssim loss
        if self.config.decoder_ssim_alpha > 0:
            decoder_ssim_loss = self.criterion_ssim(decoder_output, mel_input, output_lens)
            loss += decoder_ssim_loss * self.postnet_ssim_alpha
            return_dict["decoder_ssim_loss"] = decoder_ssim_loss

        # postnet ssim loss
        if self.config.postnet_ssim_alpha > 0:
            postnet_ssim_loss = self.criterion_ssim(postnet_output, postnet_target, output_lens)
            loss += postnet_ssim_loss * self.postnet_ssim_alpha
            return_dict["postnet_ssim_loss"] = postnet_ssim_loss

        return_dict["loss"] = loss
        return return_dict


class StyleTacotronLoss(torch.nn.Module):
    """Collection of Style Tacotron set-up based on provided config."""

    def __init__(self, c, ga_sigma=0.4):
        super().__init__()
        self.stopnet_pos_weight = c.stopnet_pos_weight
        self.ga_alpha = c.ga_alpha
        self.decoder_diff_spec_alpha = c.decoder_diff_spec_alpha
        self.postnet_diff_spec_alpha = c.postnet_diff_spec_alpha
        self.decoder_alpha = c.decoder_loss_alpha
        self.postnet_alpha = c.postnet_loss_alpha
        self.decoder_ssim_alpha = c.decoder_ssim_alpha
        self.postnet_ssim_alpha = c.postnet_ssim_alpha
        self.style_encoder_config = c.style_encoder_config
        self.config = c

        # postnet and decoder loss
        if c.loss_masking:
            self.criterion = L1LossMasked(c.seq_len_norm) if c.model in ["Tacotron"] else MSELossMasked(c.seq_len_norm)
        else:
            self.criterion = nn.L1Loss() if c.model in ["Tacotron"] else nn.MSELoss()
        # guided attention loss
        if c.ga_alpha > 0:
            self.criterion_ga = GuidedAttentionLoss(sigma=ga_sigma)
        # differential spectral loss
        if c.postnet_diff_spec_alpha > 0 or c.decoder_diff_spec_alpha > 0:
            self.criterion_diff_spec = DifferentailSpectralLoss(loss_func=self.criterion)
        # ssim loss
        if c.postnet_ssim_alpha > 0 or c.decoder_ssim_alpha > 0:
            self.criterion_ssim = SSIMLoss()
        # stopnet loss
        # pylint: disable=not-callable
        self.criterion_st = BCELossMasked(pos_weight=torch.tensor(self.stopnet_pos_weight)) if c.stopnet else None
        # style loss
        if self.style_encoder_config.se_type == 'diffusion':
            self.criterion_se = DiffusionStyleEncoderLoss(self.style_encoder_config)
        if self.style_encoder_config.se_type == 'vae':
            self.criterion_se = VAEStyleEncoderLoss(self.style_encoder_config)
        if self.style_encoder_config.se_type == 'vaeflow':
            self.criterion_se = VAEFlowStyleEncoderLoss(self.style_encoder_config)
        if self.style_encoder_config.se_type == 'vqvae':
            self.criterion_se = VQVAEStyleEncoderLoss(self.style_encoder_config)
    
    
    def forward(
        self,
        postnet_output,
        decoder_output,
        mel_input,
        linear_input,
        style_encoder_output,
        stopnet_output,
        stopnet_target,
        stop_target_length,
        output_lens,
        decoder_b_output,
        alignments,
        alignment_lens,
        alignments_backwards,
        input_lens,
    ):

        # decoder outputs linear or mel spectrograms for Tacotron and Tacotron2
        # the target should be set acccordingly
        postnet_target = linear_input if self.config.model.lower() in ["tacotron"] else mel_input

        return_dict = {}
        # remove lengths if no masking is applied
        if not self.config.loss_masking:
            output_lens = None
        # decoder and postnet losses
        if self.config.loss_masking:
            if self.decoder_alpha > 0:
                decoder_loss = self.criterion(decoder_output, mel_input, output_lens)
            if self.postnet_alpha > 0:
                postnet_loss = self.criterion(postnet_output, postnet_target, output_lens)
        else:
            if self.decoder_alpha > 0:
                decoder_loss = self.criterion(decoder_output, mel_input)
            if self.postnet_alpha > 0:
                postnet_loss = self.criterion(postnet_output, postnet_target)
        loss = self.decoder_alpha * decoder_loss + self.postnet_alpha * postnet_loss
        return_dict["decoder_loss"] = decoder_loss
        return_dict["postnet_loss"] = postnet_loss

        stop_loss = (
            self.criterion_st(stopnet_output, stopnet_target, stop_target_length)
            if self.config.stopnet
            else torch.zeros(1)
        )
        loss += stop_loss
        return_dict["stopnet_loss"] = stop_loss

        # backward decoder loss (if enabled)
        if self.config.bidirectional_decoder:
            if self.config.loss_masking:
                decoder_b_loss = self.criterion(torch.flip(decoder_b_output, dims=(1,)), mel_input, output_lens)
            else:
                decoder_b_loss = self.criterion(torch.flip(decoder_b_output, dims=(1,)), mel_input)
            decoder_c_loss = torch.nn.functional.l1_loss(torch.flip(decoder_b_output, dims=(1,)), decoder_output)
            loss += self.decoder_alpha * (decoder_b_loss + decoder_c_loss)
            return_dict["decoder_b_loss"] = decoder_b_loss
            return_dict["decoder_c_loss"] = decoder_c_loss

        # double decoder consistency loss (if enabled)
        if self.config.double_decoder_consistency:
            if self.config.loss_masking:
                decoder_b_loss = self.criterion(decoder_b_output, mel_input, output_lens)
            else:
                decoder_b_loss = self.criterion(decoder_b_output, mel_input)
            # decoder_c_loss = torch.nn.functional.l1_loss(decoder_b_output, decoder_output)
            attention_c_loss = torch.nn.functional.l1_loss(alignments, alignments_backwards)
            loss += self.decoder_alpha * (decoder_b_loss + attention_c_loss)
            return_dict["decoder_coarse_loss"] = decoder_b_loss
            return_dict["decoder_ddc_loss"] = attention_c_loss

        # guided attention loss (if enabled)
        if self.config.ga_alpha > 0:
            ga_loss = self.criterion_ga(alignments, input_lens, alignment_lens)
            loss += ga_loss * self.ga_alpha
            return_dict["ga_loss"] = ga_loss

        # decoder differential spectral loss
        if self.config.decoder_diff_spec_alpha > 0:
            decoder_diff_spec_loss = self.criterion_diff_spec(decoder_output, mel_input, output_lens)
            loss += decoder_diff_spec_loss * self.decoder_diff_spec_alpha
            return_dict["decoder_diff_spec_loss"] = decoder_diff_spec_loss

        # postnet differential spectral loss
        if self.config.postnet_diff_spec_alpha > 0:
            postnet_diff_spec_loss = self.criterion_diff_spec(postnet_output, postnet_target, output_lens)
            loss += postnet_diff_spec_loss * self.postnet_diff_spec_alpha
            return_dict["postnet_diff_spec_loss"] = postnet_diff_spec_loss

        # decoder ssim loss
        if self.config.decoder_ssim_alpha > 0:
            decoder_ssim_loss = self.criterion_ssim(decoder_output, mel_input, output_lens)
            loss += decoder_ssim_loss * self.postnet_ssim_alpha
            return_dict["decoder_ssim_loss"] = decoder_ssim_loss

        # postnet ssim loss
        if self.config.postnet_ssim_alpha > 0:
            postnet_ssim_loss = self.criterion_ssim(postnet_output, postnet_target, output_lens)
            loss += postnet_ssim_loss * self.postnet_ssim_alpha
            return_dict["postnet_ssim_loss"] = postnet_ssim_loss

        # style encoder loss diffusion based
        if self.style_encoder_config.se_type == 'diffusion':
            style_loss = self.criterion_se(style_encoder_output['noise_pred'], style_encoder_output['noise_target'])
            loss += style_loss * self.style_encoder_config.diff_loss_alpha
            return_dict["style_encoder_loss"] = style_loss
        
        # style encoder loss VAE based
        if self.style_encoder_config.se_type == 'vae':
            style_loss = self.criterion_se(style_encoder_output['mean'], style_encoder_output['log_var'])
            loss += style_loss * self.criterion_se.alpha_vae
            return_dict["style_encoder_loss"] = style_loss

        # style encoder loss VAE+Flow based
        if self.style_encoder_config.se_type == 'vaeflow':
            style_loss = self.criterion_se(style_encoder_output['z_0'], style_encoder_output['z_T'], style_encoder_output['mean'], style_encoder_output['log_var'])
            loss += style_loss * self.criterion_se.alpha_vae
            return_dict["style_encoder_loss"] = style_loss

        # style encoder loss VQVAE based
        if self.style_encoder_config.se_type == 'vqvae':
            style_loss = self.criterion_se(style_encoder_output['z_q_x'], style_encoder_output['z_e_x'])
            loss += style_loss
            return_dict["style_encoder_loss"] = style_loss

        return_dict["loss"] = loss
        return return_dict

class GlowTTSLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant_factor = 0.5 * math.log(2 * math.pi)

    def forward(self, z, means, scales, log_det, y_lengths, o_dur_log, o_attn_dur, x_lengths):
        return_dict = {}
        # flow loss - neg log likelihood
        pz = torch.sum(scales) + 0.5 * torch.sum(torch.exp(-2 * scales) * (z - means) ** 2)
        log_mle = self.constant_factor + (pz - torch.sum(log_det)) / (torch.sum(y_lengths) * z.shape[2])
        # duration loss - MSE
        loss_dur = torch.sum((o_dur_log - o_attn_dur) ** 2) / torch.sum(x_lengths)
        # duration loss - huber loss
        # loss_dur = torch.nn.functional.smooth_l1_loss(o_dur_log, o_attn_dur, reduction="sum") / torch.sum(x_lengths)
        return_dict["loss"] = log_mle + loss_dur
        return_dict["log_mle"] = log_mle
        return_dict["loss_dur"] = loss_dur

        # check if any loss is NaN
        for key, loss in return_dict.items():
            if torch.isnan(loss):
                raise RuntimeError(f" [!] NaN loss with {key}.")
        return return_dict


def mse_loss_custom(x, y):
    """MSE loss using the torch back-end without reduction.
    It uses less VRAM than the raw code"""
    expanded_x, expanded_y = torch.broadcast_tensors(x, y)
    return torch._C._nn.mse_loss(expanded_x, expanded_y, 0)  # pylint: disable=protected-access, c-extension-no-member


class MDNLoss(nn.Module):
    """Mixture of Density Network Loss as described in https://arxiv.org/pdf/2003.01950.pdf."""

    def forward(self, logp, text_lengths, mel_lengths):  # pylint: disable=no-self-use
        """
        Shapes:
            mu: [B, D, T]
            log_sigma: [B, D, T]
            mel_spec: [B, D, T]
        """
        B, T_seq, T_mel = logp.shape
        log_alpha = logp.new_ones(B, T_seq, T_mel) * (-1e4)
        log_alpha[:, 0, 0] = logp[:, 0, 0]
        for t in range(1, T_mel):
            prev_step = torch.cat(
                [log_alpha[:, :, t - 1 : t], functional.pad(log_alpha[:, :, t - 1 : t], (0, 0, 1, -1), value=-1e4)],
                dim=-1,
            )
            log_alpha[:, :, t] = torch.logsumexp(prev_step + 1e-4, dim=-1) + logp[:, :, t]
        alpha_last = log_alpha[torch.arange(B), text_lengths - 1, mel_lengths - 1]
        mdn_loss = -alpha_last.mean() / T_seq
        return mdn_loss  # , log_prob_matrix


class AlignTTSLoss(nn.Module):
    """Modified AlignTTS Loss.
    Computes
        - L1 and SSIM losses from output spectrograms.
        - Huber loss for duration predictor.
        - MDNLoss for Mixture of Density Network.

    All loss values are aggregated by a weighted sum of the alpha values.

    Args:
        c (dict): TTS model configuration.
    """

    def __init__(self, c):
        super().__init__()
        self.mdn_loss = MDNLoss()
        self.spec_loss = MSELossMasked(False)
        self.ssim = SSIMLoss()
        self.dur_loss = MSELossMasked(False)

        self.ssim_alpha = c.ssim_alpha
        self.dur_loss_alpha = c.dur_loss_alpha
        self.spec_loss_alpha = c.spec_loss_alpha
        self.mdn_alpha = c.mdn_alpha

    def forward(
        self, logp, decoder_output, decoder_target, decoder_output_lens, dur_output, dur_target, input_lens, phase
    ):
        # ssim_alpha, dur_loss_alpha, spec_loss_alpha, mdn_alpha = self.set_alphas(step)
        spec_loss, ssim_loss, dur_loss, mdn_loss = 0, 0, 0, 0
        if phase == 0:
            mdn_loss = self.mdn_loss(logp, input_lens, decoder_output_lens)
        elif phase == 1:
            spec_loss = self.spec_loss(decoder_output, decoder_target, decoder_output_lens)
            ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
        elif phase == 2:
            mdn_loss = self.mdn_loss(logp, input_lens, decoder_output_lens)
            spec_loss = self.spec_lossX(decoder_output, decoder_target, decoder_output_lens)
            ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
        elif phase == 3:
            dur_loss = self.dur_loss(dur_output.unsqueeze(2), dur_target.unsqueeze(2), input_lens)
        else:
            mdn_loss = self.mdn_loss(logp, input_lens, decoder_output_lens)
            spec_loss = self.spec_loss(decoder_output, decoder_target, decoder_output_lens)
            ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
            dur_loss = self.dur_loss(dur_output.unsqueeze(2), dur_target.unsqueeze(2), input_lens)
        loss = (
            self.spec_loss_alpha * spec_loss
            + self.ssim_alpha * ssim_loss
            + self.dur_loss_alpha * dur_loss
            + self.mdn_alpha * mdn_loss
        )
        return {"loss": loss, "loss_l1": spec_loss, "loss_ssim": ssim_loss, "loss_dur": dur_loss, "mdn_loss": mdn_loss}


class VitsGeneratorLoss(nn.Module):
    def __init__(self, c: Coqpit):
        super().__init__()
        self.kl_loss_alpha = c.kl_loss_alpha
        self.gen_loss_alpha = c.gen_loss_alpha
        self.feat_loss_alpha = c.feat_loss_alpha
        self.dur_loss_alpha = c.dur_loss_alpha
        self.mel_loss_alpha = c.mel_loss_alpha
        self.spk_encoder_loss_alpha = c.speaker_encoder_loss_alpha
        self.stft = TorchSTFT(
            c.audio.fft_size,
            c.audio.hop_length,
            c.audio.win_length,
            sample_rate=c.audio.sample_rate,
            mel_fmin=c.audio.mel_fmin,
            mel_fmax=c.audio.mel_fmax,
            n_mels=c.audio.num_mels,
            use_mel=True,
            do_amp_to_db=True,
        )

    @staticmethod
    def feature_loss(feats_real, feats_generated):
        loss = 0
        for dr, dg in zip(feats_real, feats_generated):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    @staticmethod
    def generator_loss(scores_fake):
        loss = 0
        gen_losses = []
        for dg in scores_fake:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    @staticmethod
    def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
        """
        z_p, logs_q: [b, h, t_t]
        m_p, logs_p: [b, h, t_t]
        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l

    @staticmethod
    def cosine_similarity_loss(gt_spk_emb, syn_spk_emb):
        l = -torch.nn.functional.cosine_similarity(gt_spk_emb, syn_spk_emb).mean()
        return l

    def forward(
        self,
        waveform,
        waveform_hat,
        z_p,
        logs_q,
        m_p,
        logs_p,
        z_len,
        scores_disc_fake,
        feats_disc_fake,
        feats_disc_real,
        loss_duration,
        use_speaker_encoder_as_loss=False,
        gt_spk_emb=None,
        syn_spk_emb=None,
    ):
        """
        Shapes:
            - waveform : :math:`[B, 1, T]`
            - waveform_hat: :math:`[B, 1, T]`
            - z_p: :math:`[B, C, T]`
            - logs_q: :math:`[B, C, T]`
            - m_p: :math:`[B, C, T]`
            - logs_p: :math:`[B, C, T]`
            - z_len: :math:`[B]`
            - scores_disc_fake[i]: :math:`[B, C]`
            - feats_disc_fake[i][j]: :math:`[B, C, T', P]`
            - feats_disc_real[i][j]: :math:`[B, C, T', P]`
        """
        loss = 0.0
        return_dict = {}
        z_mask = sequence_mask(z_len).float()
        # compute mel spectrograms from the waveforms
        mel = self.stft(waveform)
        mel_hat = self.stft(waveform_hat)

        # compute losses
        loss_kl = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask.unsqueeze(1)) * self.kl_loss_alpha
        loss_feat = self.feature_loss(feats_disc_fake, feats_disc_real) * self.feat_loss_alpha
        loss_gen = self.generator_loss(scores_disc_fake)[0] * self.gen_loss_alpha
        loss_mel = torch.nn.functional.l1_loss(mel, mel_hat) * self.mel_loss_alpha
        loss_duration = torch.sum(loss_duration.float()) * self.dur_loss_alpha
        loss = loss_kl + loss_feat + loss_mel + loss_gen + loss_duration

        if use_speaker_encoder_as_loss:
            loss_se = self.cosine_similarity_loss(gt_spk_emb, syn_spk_emb) * self.spk_encoder_loss_alpha
            loss += loss_se
            return_dict["loss_spk_encoder"] = loss_se

        # pass losses to the dict
        return_dict["loss_gen"] = loss_gen
        return_dict["loss_kl"] = loss_kl
        return_dict["loss_feat"] = loss_feat
        return_dict["loss_mel"] = loss_mel
        return_dict["loss_duration"] = loss_duration
        return_dict["loss"] = loss
        return return_dict


class VitsDiscriminatorLoss(nn.Module):
    def __init__(self, c: Coqpit):
        super().__init__()
        self.disc_loss_alpha = c.disc_loss_alpha

    @staticmethod
    def discriminator_loss(scores_real, scores_fake):
        loss = 0
        real_losses = []
        fake_losses = []
        for dr, dg in zip(scores_real, scores_fake):
            dr = dr.float()
            dg = dg.float()
            real_loss = torch.mean((1 - dr) ** 2)
            fake_loss = torch.mean(dg ** 2)
            loss += real_loss + fake_loss
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())

        return loss, real_losses, fake_losses

    def forward(self, scores_disc_real, scores_disc_fake):
        loss = 0.0
        return_dict = {}
        loss_disc, _, _ = self.discriminator_loss(scores_disc_real, scores_disc_fake)
        return_dict["loss_disc"] = loss_disc * self.disc_loss_alpha
        loss = loss + return_dict["loss_disc"]
        return_dict["loss"] = loss
        return return_dict


class ForwardTTSLoss(nn.Module):
    """Generic configurable ForwardTTS loss."""

    def __init__(self, c):
        super().__init__()
        if c.spec_loss_type == "mse":
            self.spec_loss = MSELossMasked(False)
        elif c.spec_loss_type == "l1":
            self.spec_loss = L1LossMasked(False)
        else:
            raise ValueError(" [!] Unknown spec_loss_type {}".format(c.spec_loss_type))

        if c.duration_loss_type == "mse":
            self.dur_loss = MSELossMasked(False)
        elif c.duration_loss_type == "l1":
            self.dur_loss = L1LossMasked(False)
        elif c.duration_loss_type == "huber":
            self.dur_loss = Huber()
        else:
            raise ValueError(" [!] Unknown duration_loss_type {}".format(c.duration_loss_type))

        if c.model_args.use_aligner:
            self.aligner_loss = ForwardSumLoss()
            self.aligner_loss_alpha = c.aligner_loss_alpha

        if c.model_args.use_pitch:
            self.pitch_loss = MSELossMasked(False)
            self.pitch_loss_alpha = c.pitch_loss_alpha

        if c.model_args.use_energy:
            self.energy_loss = MSELossMasked(False)
            self.energy_loss_alpha = c.energy_loss_alpha

        if c.use_ssim_loss:
            self.ssim = SSIMLoss() if c.use_ssim_loss else None
            self.ssim_loss_alpha = c.ssim_loss_alpha

        self.spec_loss_alpha = c.spec_loss_alpha
        self.dur_loss_alpha = c.dur_loss_alpha
        self.binary_alignment_loss_alpha = c.binary_align_loss_alpha

    @staticmethod
    def _binary_alignment_loss(alignment_hard, alignment_soft):
        """Binary loss that forces soft alignments to match the hard alignments as
        explained in `https://arxiv.org/pdf/2108.10447.pdf`.
        """
        log_sum = torch.log(torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)).sum()
        return -log_sum / alignment_hard.sum()

    def forward(
        self,
        decoder_output,
        decoder_target,
        decoder_output_lens,
        dur_output,
        dur_target,
        pitch_output,
        pitch_target,
        energy_output,
        energy_target,
        input_lens,
        alignment_logprob=None,
        alignment_hard=None,
        alignment_soft=None,
    ):
        loss = 0
        return_dict = {}
        if hasattr(self, "ssim_loss") and self.ssim_loss_alpha > 0:
            ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
            loss = loss + self.ssim_loss_alpha * ssim_loss
            return_dict["loss_ssim"] = self.ssim_loss_alpha * ssim_loss

        if self.spec_loss_alpha > 0:
            spec_loss = self.spec_loss(decoder_output, decoder_target, decoder_output_lens)
            loss = loss + self.spec_loss_alpha * spec_loss
            return_dict["loss_spec"] = self.spec_loss_alpha * spec_loss

        if self.dur_loss_alpha > 0:
            log_dur_tgt = torch.log(dur_target.float() + 1)
            dur_loss = self.dur_loss(dur_output[:, :, None], log_dur_tgt[:, :, None], input_lens)
            loss = loss + self.dur_loss_alpha * dur_loss
            return_dict["loss_dur"] = self.dur_loss_alpha * dur_loss

        if hasattr(self, "pitch_loss") and self.pitch_loss_alpha > 0:
            pitch_loss = self.pitch_loss(pitch_output.transpose(1, 2), pitch_target.transpose(1, 2), input_lens)
            loss = loss + self.pitch_loss_alpha * pitch_loss
            return_dict["loss_pitch"] = self.pitch_loss_alpha * pitch_loss

        if hasattr(self, "energy_loss") and self.energy_loss_alpha > 0:
            energy_loss = self.energy_loss(energy_output.transpose(1, 2), energy_target.transpose(1, 2), input_lens)
            loss = loss + self.energy_loss_alpha * energy_loss
            return_dict["loss_energy"] = self.energy_loss_alpha * energy_loss

        if hasattr(self, "aligner_loss") and self.aligner_loss_alpha > 0:
            aligner_loss = self.aligner_loss(alignment_logprob, input_lens, decoder_output_lens)
            loss = loss + self.aligner_loss_alpha * aligner_loss
            return_dict["loss_aligner"] = self.aligner_loss_alpha * aligner_loss

        if self.binary_alignment_loss_alpha > 0 and alignment_hard is not None:
            binary_alignment_loss = self._binary_alignment_loss(alignment_hard, alignment_soft)
            loss = loss + self.binary_alignment_loss_alpha * binary_alignment_loss
            return_dict["loss_binary_alignment"] = self.binary_alignment_loss_alpha * binary_alignment_loss

        return_dict["loss"] = loss
        return return_dict

# StyleForwardTTS loss
class StyleForwardTTSLoss(nn.Module):
    """Generic configurable StyleForwardTTS loss."""

    def __init__(self, c):
        super().__init__()

        self.style_encoder_config = c.style_encoder_config
        self.config = c

        # Spectrogram Loss
        if c.spec_loss_type == "mse":
            self.spec_loss = MSELossMasked(False)
        elif c.spec_loss_type == "l1":
            self.spec_loss = L1LossMasked(False)
        else:
            raise ValueError(" [!] Unknown spec_loss_type {}".format(c.spec_loss_type))
        self.spec_loss_alpha = c.spec_loss_alpha
        
        # Structural Similarity Loss
        if c.use_ssim_loss:
            self.ssim = SSIMLoss() if c.use_ssim_loss else None
            self.ssim_loss_alpha = c.ssim_loss_alpha
            
        # Duration Loss
        if c.duration_loss_type == "mse":
            self.dur_loss = MSELossMasked(False)
        elif c.duration_loss_type == "l1":
            self.dur_loss = L1LossMasked(False)
        elif c.duration_loss_type == "huber":
            self.dur_loss = Huber()
        else:
            raise ValueError(" [!] Unknown duration_loss_type {}".format(c.duration_loss_type))
        self.dur_loss_alpha = c.dur_loss_alpha
        
        # Aligner Loss
        if c.model_args.use_aligner:
            self.aligner_loss = ForwardSumLoss()
            self.aligner_loss_alpha = c.aligner_loss_alpha
        self.binary_alignment_loss_alpha = c.binary_align_loss_alpha
        
        # Pitch Loss
        if c.model_args.use_pitch:
            self.pitch_loss = MSELossMasked(False)
            self.pitch_loss_alpha = c.pitch_loss_alpha

        # Energy Loss
        if c.model_args.use_energy:
            self.energy_loss = MSELossMasked(False)
            self.energy_loss_alpha = c.energy_loss_alpha

        # Style Encoder Achitecture Specific Loss
        if self.style_encoder_config.se_type == 'diffusion':
            self.criterion_se = DiffusionStyleEncoderLoss(self.style_encoder_config)
        if self.style_encoder_config.se_type == 'vae':
            self.criterion_se = VAEStyleEncoderLoss(self.style_encoder_config)
        if self.style_encoder_config.se_type == 'vaeflow':
            self.criterion_se = VAEFlowStyleEncoderLoss(self.style_encoder_config)
        if self.style_encoder_config.se_type == 'vqvae':
            self.criterion_se = VQVAEStyleEncoderLoss(self.style_encoder_config)

        # Style Classifier Loss
        if self.style_encoder_config.use_guided_style:
            if self.style_encoder_config.guided_style_loss_type == 'ce':
                self.criterion_guided = nn.CrossEntropyLoss()
            elif self.style_encoder_config.guided_style_loss_type == 'focal_loss':
                self.criterion_guided = BalancedCrossEntropyLoss(loss_type='focal_loss')
            else:
                raise NotImplementedError
        
        # Speaker Classifier w/ GRL Loss
        if self.style_encoder_config.use_grl_on_speakers_in_style_embedding:
            if self.style_encoder_config.grl_loss_type == 'ce':
                self.criterion_grl_speaker_in_style_embedding = nn.CrossEntropyLoss()
            elif self.style_encoder_config.grl_loss_type == 'focal_loss':
                self.criterion_grl_speaker_in_style_embedding = BalancedCrossEntropyLoss(loss_type='focal_loss')
            else:
                raise NotImplementedError
            self.grl_speaker_in_style_embedding_alpha = self.style_encoder_config.grl_alpha
            
        # Residual Speaker Disentanglement Loss
        if self.style_encoder_config.use_residual_speaker_disentanglement:
            if self.style_encoder_config.grl_loss_type == 'ce':
                self.criterion_speaker_guided = nn.CrossEntropyLoss()
                self.criterion_grl_style_in_speaker_embedding = nn.CrossEntropyLoss()
            elif self.style_encoder_config.grl_loss_type == 'focal_loss':
                self.criterion_speaker_guided = BalancedCrossEntropyLoss(loss_type='focal_loss')
                self.criterion_grl_style_in_speaker_embedding = BalancedCrossEntropyLoss(loss_type='focal_loss')
            else:
                raise NotImplementedError
                
        # CLIP Loss
        if(self.style_encoder_config.use_clip_loss):
            self.criterion_clip = ClipLloss(self.style_encoder_config)

        # Style Distortion Loss
        if(self.style_encoder_config.use_style_distortion_loss):
            self.criterion_style_distortion = nn.MSELoss() # Simple L2 norm

        # Cycle Consistency Loss
        if(self.style_encoder_config.use_cycle_consistency):
            self.criterion_cycle_consistency = nn.MSELoss()
            self.criterion_speaker_cycle_consistency = nn.MSELoss()

    @staticmethod
    def _binary_alignment_loss(alignment_hard, alignment_soft):
        """Binary loss that forces soft alignments to match the hard alignments as
        explained in `https://arxiv.org/pdf/2108.10447.pdf`.
        """
        log_sum = torch.log(torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)).sum()
        return -log_sum / alignment_hard.sum()

    def forward(
        self,
        decoder_output,
        decoder_target,
        decoder_output_lens,
        dur_output,
        dur_target,
        pitch_output,
        pitch_target,
        energy_output,
        energy_target,
        input_lens,
        style_encoder_output,
        alignment_logprob=None,
        alignment_hard=None,
        alignment_soft=None,
        style_ids=None,
        speaker_ids = None,
        encoder_output=None,
        speaker_output=None,
        style_preds=None,
        speaker_preds_from_style=None,
        ressynt_style_encoder_output=None,
        cycle_style_encoder_output=None,
        speaker_embeddings_cycle= None,
        step = None,
        residual_speaker_embeddings = None,
        residual_style_preds = None,
        residual_speaker_preds = None
    ):
        loss = 0
        return_dict = {}
        if hasattr(self, "ssim_loss") and self.ssim_loss_alpha > 0:
            ssim_loss = self.ssim(decoder_output, decoder_target, decoder_output_lens)
            loss = loss + self.ssim_loss_alpha * ssim_loss
            return_dict["loss_ssim"] = self.ssim_loss_alpha * ssim_loss

        if self.spec_loss_alpha > 0:
            spec_loss = self.spec_loss(decoder_output, decoder_target, decoder_output_lens)
            loss = loss + self.spec_loss_alpha * spec_loss
            return_dict["loss_spec"] = self.spec_loss_alpha * spec_loss

        if self.dur_loss_alpha > 0:
            log_dur_tgt = torch.log(dur_target.float() + 1)
            dur_loss = self.dur_loss(dur_output[:, :, None], log_dur_tgt[:, :, None], input_lens)
            loss = loss + self.dur_loss_alpha * dur_loss
            return_dict["loss_dur"] = self.dur_loss_alpha * dur_loss

        if hasattr(self, "pitch_loss") and self.pitch_loss_alpha > 0:
            pitch_loss = self.pitch_loss(pitch_output.transpose(1, 2), pitch_target.transpose(1, 2), input_lens)
            loss = loss + self.pitch_loss_alpha * pitch_loss
            return_dict["loss_pitch"] = self.pitch_loss_alpha * pitch_loss
        
        if hasattr(self, "energy_loss") and self.energy_loss_alpha > 0:
            energy_loss = self.energy_loss(energy_output.transpose(1, 2), energy_target.transpose(1, 2), input_lens)
            loss = loss + self.energy_loss_alpha * energy_loss
            return_dict["loss_energy"] = self.energy_loss_alpha * energy_loss

        if hasattr(self, "aligner_loss") and self.aligner_loss_alpha > 0:
            aligner_loss = self.aligner_loss(alignment_logprob, input_lens, decoder_output_lens)
            loss = loss + self.aligner_loss_alpha * aligner_loss
            return_dict["loss_aligner"] = self.aligner_loss_alpha * aligner_loss

        if self.binary_alignment_loss_alpha > 0 and alignment_hard is not None:
            binary_alignment_loss = self._binary_alignment_loss(alignment_hard, alignment_soft)
            loss = loss + self.binary_alignment_loss_alpha * binary_alignment_loss
            return_dict["loss_binary_alignment"] = self.binary_alignment_loss_alpha * binary_alignment_loss

        # style encoder loss diffusion based
        if self.style_encoder_config.se_type == 'diffusion':
            style_loss = self.criterion_se(style_encoder_output['noises']['noise_pred'], style_encoder_output['noises']['noise_target'])
            loss += style_loss * self.style_encoder_config.diff_loss_alpha
            return_dict["style_encoder_loss"] = style_loss
        
        # style encoder loss VAE based
        if self.style_encoder_config.se_type == 'vae':
            if(step is not None):
                self.criterion_se.step = step
            style_loss = self.criterion_se(style_encoder_output['mean'], style_encoder_output['log_var'], step)
            loss += style_loss * self.criterion_se.alpha_vae
            return_dict["style_encoder_loss"] = style_loss
            return_dict['alpha_value'] = self.criterion_se.alpha_vae

            if self.style_encoder_config.use_guided_style:
                # print(style_preds.shape, style_ids.shape)
                style_guided_loss = self.criterion_guided(style_preds, style_ids) # Must squeeze cuz it was augmented for broadcasting
                
                # print(style_guided_loss, torch.argmax(style_preds, axis = -1), style_ids)
                
                loss += style_guided_loss
                return_dict["style_guided_loss"] = style_guided_loss

            if self.style_encoder_config.use_grl_on_speakers_in_style_embedding:
                grl_speaker_in_style_loss = self.criterion_grl_speaker_in_style_embedding(speaker_preds_from_style, speaker_ids)
                
                loss += self.grl_speaker_in_style_embedding_alpha*grl_speaker_in_style_loss
                
                return_dict["grl_speaker_in_style_embedding"] = grl_speaker_in_style_loss

        # style encoder loss VQVAE based
        if self.style_encoder_config.se_type == 'vqvae':
            loss_vq, loss_commit = self.criterion_se(style_encoder_output['z_q'], style_encoder_output['z_e'])
            style_loss = loss_vq + self.criterion_se.beta_vqvae * loss_commit
            loss += style_loss
            return_dict["loss_vq"] = loss_vq
            return_dict["loss_commit"] = loss_commit
            return_dict["style_loss"] = style_loss

            if self.style_encoder_config.use_guided_style:
                style_guided_loss = self.criterion_guided(style_preds, style_ids) # Must squeeze cuz it was augmented for broadcasting                
                loss += style_guided_loss
                return_dict["style_guided_loss"] = style_guided_loss

            if self.style_encoder_config.use_grl_on_speakers_in_style_embedding:
                grl_speaker_in_style_loss = self.criterion_grl_speaker_in_style_embedding(speaker_preds_from_style, speaker_ids)
                loss += self.grl_speaker_in_style_embedding_alpha*grl_speaker_in_style_loss
                return_dict["grl_speaker_in_style_embedding"] = grl_speaker_in_style_loss

        # style encoder loss VAE based
        if self.style_encoder_config.se_type == 'vaeflow':
            style_loss = self.criterion_se(style_encoder_output['z_0'], style_encoder_output['z_T'], style_encoder_output['mean'], style_encoder_output['log_var'])
            loss += style_loss * self.criterion_se.alpha_vae
            return_dict["style_encoder_loss"] = style_loss

        if(self.style_encoder_config.se_type == 're' or self.style_encoder_config.se_type == 'modifiedre'):
            if self.style_encoder_config.content_orthogonal_loss:
                # print(style_encoder_output.shape, encoder_output.shape)
                # print(torch.matmul(style_encoder_output, encoder_output.permute(0,2,1)))
                # content_dot_loss = torch.dot(style_encoder_output, encoder_output)
                content_dot_loss = (style_encoder_output['style_embedding'].unsqueeze(1)*encoder_output.permute(0,2,1)).sum(axis=-1).mean().abs() # Must be abs because dot product can be negative
                # print(content_dot_loss)
                loss += content_dot_loss
                return_dict["content_orthogonal_loss"] = content_dot_loss
            if self.style_encoder_config.speaker_orthogonal_loss:
                # print(style_encoder_output.shape, speaker_output.shape)
                speaker_dot_loss = torch.inner(style_encoder_output['style_embedding'], speaker_output.squeeze(-1)).mean().abs()
                # print(torch.inner(style_encoder_output, speaker_output.squeeze(-1)).shape)
                loss += speaker_dot_loss
                return_dict["speaker_orthogonal_loss"] = speaker_dot_loss

        if self.style_encoder_config.use_guided_style:
            # print(style_preds.shape, style_ids.shape)
            style_guided_loss = self.criterion_guided(style_preds, style_ids) # Must squeeze cuz it was augmented for broadcasting
            
            # print(style_guided_loss, torch.argmax(style_preds, axis = -1), style_ids)
            
            loss += style_guided_loss
            return_dict["style_guided_loss"] = style_guided_loss

        if self.style_encoder_config.use_grl_on_speakers_in_style_embedding:
    
            grl_speaker_in_style_loss = torch.clamp(self.criterion_grl_speaker_in_style_embedding(speaker_preds_from_style, speaker_ids), min= 0, max = 4)
            
            loss += self.grl_speaker_in_style_embedding_alpha*grl_speaker_in_style_loss
            
            return_dict["grl_speaker_in_style_embedding"] = grl_speaker_in_style_loss




        if(self.style_encoder_config.use_clip_loss):
            # print(style_encoder_output['style_embedding'].shape)
            # print(ressynt_style_encoder_output['style_embedding'].shape)
            if(self.style_encoder_config.se_type == 'vae'):
                clip_loss = self.criterion_clip(style_encoder_output['style_embedding'].squeeze(1), ressynt_style_encoder_output.squeeze(1))
            else:
                clip_loss = self.criterion_clip(style_encoder_output['style_embedding'], ressynt_style_encoder_output)
            loss += clip_loss #It already has the alpha in it

            return_dict['clip_loss'] = clip_loss
        
        if(self.style_encoder_config.use_style_distortion_loss):
            # print(style_encoder_output['style_embedding'].shape)
            # print(ressynt_style_encoder_output['style_embedding'].shape)
            if(self.style_encoder_config.se_type == 'vae'):
                style_distortion_loss = self.criterion_style_distortion(style_encoder_output['style_embedding'].squeeze(1), ressynt_style_encoder_output.squeeze(1))
            else:
                style_distortion_loss = self.criterion_style_distortion(style_encoder_output['style_embedding'], ressynt_style_encoder_output)
            loss += style_distortion_loss*self.style_encoder_config.style_distortion_alpha_loss

            return_dict['identity_style_loss'] = style_distortion_loss

        if(self.style_encoder_config.use_cycle_consistency):
            
            if(self.style_encoder_config.start_cycle_at < step):

                cycle_style_loss = self.criterion_cycle_consistency(style_encoder_output['style_embedding'], cycle_style_encoder_output)

                loss += cycle_style_loss*self.style_encoder_config.cycle_consistency_alpha

                return_dict['cycle_style_distortion_loss'] = cycle_style_loss
        
            else:

                return_dict['cycle_style_distortion_loss'] = 9999 # Just a number that you will see in trainer log output and confirm that this loss is not being propagated

        if self.style_encoder_config.use_residual_speaker_disentanglement:
            
            batch_len = residual_speaker_embeddings.shape[0]

            speaker_guided_loss = self.criterion_speaker_guided(residual_speaker_preds, speaker_ids)

            grl_style_in_speaker_loss = torch.clamp(self.criterion_grl_style_in_speaker_embedding(residual_style_preds, style_ids), min = 0, max = 4)

            speaker_style_orthogonal_loss = torch.trace(torch.inner(style_encoder_output['style_embedding'], residual_speaker_embeddings).abs())/batch_len

            loss += (speaker_guided_loss + grl_style_in_speaker_loss + speaker_style_orthogonal_loss)
        
            return_dict['speaker_guided_loss'] = speaker_guided_loss 
            return_dict['grl_style_in_speaker_embedding'] = grl_style_in_speaker_loss
            return_dict['speaker_style_ortho_loss'] = speaker_style_orthogonal_loss

            if(self.style_encoder_config.use_cycle_consistency):
                cycle_speaker_loss = self.criterion_speaker_cycle_consistency(speaker_embeddings_cycle, residual_speaker_embeddings)
                return_dict['cycle_speaker_distortion_loss'] = cycle_speaker_loss

                loss += cycle_speaker_loss
        # Just for checking losses, todo: turn this optional
        return_dict['which_step_in_model_loss_calc'] = step
        # return_dict['NOT_OPT_identity_style_loss'] = nn.MSELoss()(style_encoder_output['style_embedding'], ressynt_style_encoder_output)

        return_dict["loss"] = loss
        return return_dict
