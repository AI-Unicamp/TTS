import copy
from abc import abstractmethod
from typing import Dict

import torch
from coqpit import Coqpit
from torch import nn

from TTS.tts.layers.losses import StyleTacotronLoss
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import sequence_mask
from TTS.utils.generic_utils import format_aux_input
from TTS.utils.io import load_fsspec
from TTS.utils.training import gradual_training_scheduler


class BaseTacotron(BaseTTS):
    """Base class shared by StyleTacotrons"""

    def __init__(self, config: Coqpit):
        super().__init__(config)

        # pass all config fields as class attributes
        for key in config:
            setattr(self, key, config[key])

        # layers
        self.embedding = None
        self.encoder = None
        self.decoder = None
        self.postnet = None

        # init tensors
        self.embedded_speakers = None
        self.embedded_speakers_projected = None

        # init style layers
        if(self.style_encoder_config.agg_type == 'concat'):
            self.decoder_in_features += self.style_encoder_config.style_embedding_dim 

        self.style_encoder_layer = None

        # additional layers
        self.decoder_backward = None
        self.coarse_decoder = None

    @staticmethod
    def _format_aux_input(aux_input: Dict) -> Dict:
        """Set missing fields to their default values"""
        if aux_input:
            return format_aux_input({"d_vectors": None, "speaker_ids": None}, aux_input)
        return None

    #############################
    # INIT FUNCTIONS
    #############################

    def _init_backward_decoder(self):
        """Init the backward decoder for Forward-Backward decoding."""
        self.decoder_backward = copy.deepcopy(self.decoder)

    def _init_coarse_decoder(self):
        """Init the coarse decoder for Double-Decoder Consistency."""
        self.coarse_decoder = copy.deepcopy(self.decoder)
        self.coarse_decoder.r_init = self.ddc_r
        self.coarse_decoder.set_r(self.ddc_r)

    #############################
    # CORE FUNCTIONS
    #############################

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load model checkpoint and set up internals.

        Args:
            config (Coqpi): model configuration.
            checkpoint_path (str): path to checkpoint file.
            eval (bool): whether to load model for evaluation.
        """
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        # TODO: set r in run-time by taking it from the new config
        if "r" in state:
            # set r from the state (for compatibility with older checkpoints)
            self.decoder.set_r(state["r"])
        elif "config" in state:
            # set r from config used at training time (for inference)
            self.decoder.set_r(state["config"]["r"])
        else:
            # set r from the new config (for new-models)
            self.decoder.set_r(config.r)
        if eval:
            self.eval()
            print(f" > Model's reduction rate `r` is set to: {self.decoder.r}")
            assert not self.training

    def get_criterion(self) -> nn.Module:
        """Get the model criterion used in training."""
        return StyleTacotronLoss(self.config)

    #############################
    # COMMON COMPUTE FUNCTIONS
    #############################

    def compute_masks(self, text_lengths, mel_lengths):
        """Compute masks  against sequence paddings."""
        # B x T_in_max (boolean)
        input_mask = sequence_mask(text_lengths)
        output_mask = None
        if mel_lengths is not None:
            max_len = mel_lengths.max()
            r = self.decoder.r
            max_len = max_len + (r - (max_len % r)) if max_len % r > 0 else max_len
            output_mask = sequence_mask(mel_lengths, max_len=max_len)
        return input_mask, output_mask

    def _backward_pass(self, mel_specs, encoder_outputs, mask):
        """Run backwards decoder"""
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask
        )
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2).contiguous()
        return decoder_outputs_b, alignments_b

    def _coarse_decoder_pass(self, mel_specs, encoder_outputs, alignments, input_mask):
        """Double Decoder Consistency"""
        T = mel_specs.shape[1]
        if T % self.coarse_decoder.r > 0:
            padding_size = self.coarse_decoder.r - (T % self.coarse_decoder.r)
            mel_specs = torch.nn.functional.pad(mel_specs, (0, 0, 0, padding_size, 0, 0))
        decoder_outputs_backward, alignments_backward, _ = self.coarse_decoder(
            encoder_outputs.detach(), mel_specs, input_mask
        )
        # scale_factor = self.decoder.r_init / self.decoder.r
        alignments_backward = torch.nn.functional.interpolate(
            alignments_backward.transpose(1, 2), size=alignments.shape[1], mode="nearest"
        ).transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward.transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward[:, :T, :]
        return decoder_outputs_backward, alignments_backward

    #############################
    # EMBEDDING FUNCTIONS
    #############################

    @staticmethod
    def _add_speaker_embedding(outputs, embedded_speakers):
        embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = outputs + embedded_speakers_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, embedded_speakers):
        embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, embedded_speakers_], dim=-1)
        return outputs

    #############################
    # CALLBACKS
    #############################

    def on_epoch_start(self, trainer):
        """Callback for setting values wrt gradual training schedule.

        Args:
            trainer (TrainerTTS): TTS trainer object that is used to train this model.
        """
        if self.gradual_training:
            r, trainer.config.batch_size = gradual_training_scheduler(trainer.total_steps_done, trainer.config)
            trainer.config.r = r
            self.decoder.set_r(r)
            if trainer.config.bidirectional_decoder:
                trainer.model.decoder_backward.set_r(r)
            print(f"\n > Number of output frames: {self.decoder.r}")
