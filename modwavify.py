"""
modwavify.py

A Python commandline tool to convert wavetables exported from Vital/Vitalium to a format suitable for the Korg modwave. Written with no platform-specific code, so it should run fine in Windows, MacOS, and Linux.

The modwave support 16-bit integer and 32-bit float samples. modwavify converts any .wav file to the supported 32-bit float format:
- float32 samples
- 2048 samples per wave
- *up to* 64 waves per wavetable

Vital exports 16-bit integer, 2048 sample-per-wave, 256 wave-per-wavetable .wav files, so modwavify's default behaviour is to convert each individual sample to a float32 and average every 4-wave group into a single wave.

Modwavify should accept and transform arbitrary .wav files to wavetables.

Author: Gary Ruben
License: MIT

"""
from pathlib import Path
import click
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def waterfall(filepath, wavetable):
    """Creates and saves a png thumbnail of the wavetable

    Args:
        filepath (Path): path object with png filename
        wavetable (ndarray): n x 64 float32 ndarray of wavetable data

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(
        box_aspect=(1, 1, 0.1),
        yticklabels=[],
        zticklabels=[],
        yticks=[],
        zticks=[],
        xmargin=0,
        ymargin=0,
        frame_on=False,
    )
    ax.azim = 200
    ax.elev = 35
    ax.autoscale(tight=True)
    ax.grid(visible=False)
    plt.title(filepath.stem, y=0.9, fontsize=30)

    wave_count, wavelength = wavetable.shape
    X, Y = np.mgrid[:wave_count, :wavelength]

    ax.plot_wireframe(X, Y, wavetable, cstride=wavelength, lw=0.5, color='k')
    fig.subplots_adjust(top=1.1, bottom=-0.15, left=-0.1, right=1.1)
    plt.savefig(filepath)


@click.command()
@click.option('--average', '-a', is_flag=True, default=False, help='Chooses waves from input by local-average instead of by stride.')
@click.option('--local/--no-local', default=True, help='Normalize each wave to local (default) or global max.')
@click.option('--flip', '-f', is_flag=True, default=False, help='Reverses the wave order in the wavetable.')
@click.option('--crop', '-c', is_flag=True, default=False, help='Crop to first 64 waves instead of averaging.')
@click.option('--wavelength', '-w', default=2048, type=int, help='Samples-per-wave: 2048 (default).')
@click.option('--no-thumb', '-n', is_flag=True, default=False, help='Set this to skip the thumbnail plot.')
@click.argument('filepath', type=click.Path(exists=True))
def main(filepath, average, local, flip, crop, wavelength, no_thumb):
    """Downsamples a 2048-sample-per-wave wavetable to be modwave compatible."""

    x, _Fs = sf.read(filepath)
    if len(x.shape) > 1:
        # stereo inout; merge channels
        x = x.sum(axis=1)

    print(f'{click.format_filename(filepath)}: {x.size} samples, Max excursion {np.amax(np.abs(x))}')

    path = Path(filepath)
    output_filepath = path.with_name(path.stem + '_mw' + path.suffix)

    # Start by truncating, if necessary, to a wave boundary; usually this is idempotent
    x = x[:int(x.size / wavelength) * wavelength]
    x.shape = (-1, wavelength)

    if crop:
        x = x[:64]
        print(f"{output_filepath}: cropped to first 64 waves")
    else:
        wave_count, _ = x.shape
        if wave_count >= 64:
            # Vital wavetables contain 256 waves. The following code should always reduce this
            # by a factor of 4, but the code supports other factors for other wavetable sources
            factor = int(wave_count / 64)
            x = x[:factor * 64]
            x.shape = (-1, factor, wavelength)
            if average:
                x = x.sum(axis=1)
            else:
                x = x[:, 0]
            wave_count, _ = x.shape
            print(f"{output_filepath}: reduced by factor {factor} to {wave_count} waves")

    if flip:
        x = x[::-1]

    if local:
        # Normalize each wave individually
        max_xs = np.amax(np.abs(x), axis=1)
        x = x / max_xs[:, np.newaxis]
        wavetable = x[:]
        x = x.flatten()
    else:
        # --no-local option: Normalize all waves to their collective maximum
        wavetable = x[:]
        x = x.flatten()
        max_x = np.amax(np.abs(x))
        x = x / max_x

    sf.write(
        output_filepath,
        x,
        samplerate=44100,
        subtype='FLOAT'
    )

    if not no_thumb:
        waterfall(path.with_name(path.stem + '.png'), wavetable)


if __name__ == "__main__":
    main()
