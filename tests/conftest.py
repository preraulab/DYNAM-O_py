"""Shared pytest fixtures for pydynamo."""

from pathlib import Path

import numpy as np
import pytest

#: The DYNAM-O_toolbox checkout this repo sits in, if any.
TOOLBOX_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def synthetic_night():
    """Twenty minutes of synthetic sleep EEG at 100 Hz: a 0.8 Hz slow
    oscillation, 13 Hz spindle bursts locked to its up-phase, pink-ish
    noise and one flat-line stretch (an artifact), staged N2/N3 with a
    wake epoch. Seeded, so every test sees the same recording."""
    rng = np.random.default_rng(20261122)
    fs = 100.0
    duration_s = 20 * 60
    t = np.arange(int(duration_s * fs)) / fs

    slow_oscillation = 40.0 * np.sin(2 * np.pi * 0.8 * t)
    spindle_envelope = np.zeros_like(t)
    for start in np.arange(20.0, duration_s - 20.0, 7.3):
        width = 0.6 + 0.4 * rng.random()
        spindle_envelope += np.exp(-0.5 * ((t - start) / (width / 2.5)) ** 2)
    up_phase = 0.5 * (1.0 + np.sin(2 * np.pi * 0.8 * t))
    spindles = 18.0 * spindle_envelope * up_phase * np.sin(2 * np.pi * 13.0 * t)
    noise = np.cumsum(rng.normal(size=t.size))
    noise = 4.0 * (noise - np.convolve(noise, np.ones(200) / 200, mode="same"))
    noise += 3.0 * rng.normal(size=t.size)
    data = slow_oscillation + spindles + noise
    data[int(600 * fs):int(612 * fs)] = 0.0  # flat line -> artifact

    stage_times = np.arange(0.0, duration_s, 30.0)
    stage_vals = np.full(stage_times.size, 2.0)
    stage_vals[10:20] = 1.0   # N3
    stage_vals[30] = 5.0      # wake
    return {"data": data, "fs": fs, "stage_times": stage_times, "stage_vals": stage_vals}
