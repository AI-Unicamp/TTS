from typing import Union

import numpy as np
import parselmouth

# from config import Config


def sampler(ratio):
  shifts = np.random.rand((1)) * (ratio - 1.) + 1.

  # print(shifts)
  # flip
  flip = np.random.rand((1)) < 0.5

  # print(flip)

  shifts[flip] = shifts[flip] ** -1
  return shifts[0]

class PraatAugment:
    """Praat based augmentation.
    """
    def __init__(self,
                 sample_rate: int,
                 pitch_steps: float = 0.01,
                 pitch_floor: float = 75,
                 pitch_ceil: float = 600):
        """Initializer.
        Args:
            sample_rate: sample_rate.
            pitch_steps: pitch measurement intervals.
            pitch_floor: minimum pitch.
            pitch_ceil: maximum pitch.
        """
        self.sample_rate = sample_rate
        self.pitch_steps = pitch_steps
        self.pitch_floor = pitch_floor
        self.pitch_ceil = pitch_ceil

    def augment(self,
                snd: Union[parselmouth.Sound, np.ndarray],
                formant_shift: float = 1.,
                pitch_shift: float = 1.,
                pitch_range: float = 1.,
                duration_factor: float = 1.) -> np.ndarray:
        """Augment the sound signal with praat.
        """

        # HARDCODED random formant shit
        formant_shift = sampler(1.4)

        if not isinstance(snd, parselmouth.Sound):
            snd = parselmouth.Sound(snd, sampling_frequency=self.sample_rate)
        pitch = parselmouth.praat.call(
            snd, 'To Pitch', self.pitch_steps, self.pitch_floor, self.pitch_ceil)
        ndpit = pitch.selected_array['frequency']
        # if all unvoiced
        nonzero = ndpit > 1e-5
        if nonzero.sum() == 0:
            return snd.values[0]
        # if voiced
        median, minp = np.median(ndpit[nonzero]).item(), ndpit[nonzero].min().item()
        # scale
        updated = median * pitch_shift
        scaled = updated + (minp * pitch_shift - updated) * pitch_range
        # for preventing infinite loop of `Change gender`
        # ref:https://github.com/praat/praat/issues/1926
        if scaled < 0.:
            pitch_range = 1.
        out, = parselmouth.praat.call(
            (snd, pitch), 'Change gender',
            formant_shift,
            median * pitch_shift,
            pitch_range,
            duration_factor).values
        return out