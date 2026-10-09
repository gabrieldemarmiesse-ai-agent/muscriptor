"""MIDI output utilities."""

import bisect
import dataclasses
import logging
import math
from collections import defaultdict

import numpy as np

from muscriptor.tokenizer.notes import (
    Note,
    note2note_event,
    note_event2midi,
    trim_overlapping_notes,
)
from muscriptor.utils.beats import BeatGrid, Onsets, estimate_onset_delay

logger = logging.getLogger(__name__)

# Written when no grid was detected: 120 BPM and no time signature, leaving the
# meter for notation software to guess.
PLACEHOLDER_GRID = BeatGrid(bpm=120, beats_per_bar=None, first_downbeat=0.0)


@dataclasses.dataclass(frozen=True)
class ExportOptions:
    """How to write a transcription out as MIDI.

    The first three fields correct the detected beat grid (None keeps what was
    detected); they move bar lines, never notes. The rest tidy the notes up for
    notation software and DAWs, which turn every few milliseconds of slop into
    a 1/64 rest.
    """

    bpm: float | None = None
    beats_per_bar: int | None = None
    # A bar line, in seconds of the original audio.
    first_downbeat: float | None = None
    # Snap note starts and ends onto a subdivision of the beat.
    quantize: bool = False
    # Grid steps per beat to snap to (4 = sixteenths, 3 = eighth triplets...);
    # None uses the one the onsets were measured to sit on.
    subdivision: int | None = None
    # How far to move the notes towards the grid: 1.0 all the way, 0.5 halfway.
    strength: float = 1.0
    # Extend a note to the next onset of its instrument when the silence in
    # between is at most this many beats. 0 leaves the note lengths alone.
    fill_gaps: float = 0.0
    # Drop the empty bars before the first note. The MIDI then no longer starts
    # where the audio does.
    trim_leading_bars: bool = False


def resolve_grid(
    beat_grid: BeatGrid | None, options: ExportOptions, onsets: Onsets
) -> BeatGrid:
    """`beat_grid` with `options`' corrections applied and its onset delay measured.

    The onset delay is measured against the detected beats before any
    correction, so it is the same whatever the user does to the grid: it is a
    property of the transcription, not of the bar lines.

    `beat_subdivision` comes back None when there is nothing to quantize to:
    no tempo at all (detected or given), or onsets that sit on no subdivision of
    the beat.
    """
    tempo_known = beat_grid is not None or options.bpm is not None
    grid = beat_grid or PLACEHOLDER_GRID
    if grid.onset_delay is None:
        grid = grid.with_onset_delay(onsets)

    retimed = options.bpm is not None and not math.isclose(options.bpm, grid.bpm)
    grid = dataclasses.replace(
        grid,
        bpm=options.bpm if options.bpm is not None else grid.bpm,
        beats_per_bar=options.beats_per_bar or grid.beats_per_bar,
        first_downbeat=(
            options.first_downbeat
            if options.first_downbeat is not None
            else grid.first_downbeat
        ),
        # Both belong to the detected tempo, and say nothing about another one.
        beats=None if retimed else grid.beats,
        beat_subdivision=None if retimed else grid.beat_subdivision,
    )

    if not tempo_known:
        return dataclasses.replace(grid, beat_subdivision=None)
    if options.subdivision is not None:
        return dataclasses.replace(grid, beat_subdivision=options.subdivision)
    if options.quantize and grid.beat_subdivision is None:
        delay = grid.onset_delay
        return dataclasses.replace(
            grid,
            beat_subdivision=measure_subdivision(
                np.asarray(onsets, dtype=float) - delay, grid
            ),
        )
    return grid


def measure_subdivision(onsets: np.ndarray, grid: BeatGrid) -> int | None:
    """The subdivision of `grid`'s beats that `onsets` sit on, if any.

    For a grid without tracked beats (a tempo typed in by the user): the beats
    are laid out at constant tempo from `first_downbeat` over the onsets' span.
    """
    if not len(onsets):
        return None
    period = 60.0 / grid.bpm
    first = math.floor((onsets.min() - grid.first_downbeat) / period)
    last = math.ceil((onsets.max() - grid.first_downbeat) / period)
    beats = grid.first_downbeat + period * np.arange(first, last + 1)
    measured = estimate_onset_delay(onsets, dataclasses.replace(grid, beats=beats))
    return measured.subdivision if measured else None


def shifted_notes(notes: list[Note], delay_s: float) -> list[Note]:
    """`notes` moved by `delay_s`. May go negative; the bar offset covers that."""
    if not delay_s:
        return notes
    return [
        dataclasses.replace(
            note, onset=note.onset + delay_s, offset=note.offset + delay_s
        )
        for note in notes
    ]


def notes_to_midi(
    notes: list[Note],
    velocity: int = 100,
    program_names: dict[int, str] | None = None,
    beat_grid: BeatGrid | None = None,
    options: ExportOptions | None = None,
):
    """Convert a list of Note objects to a mido MidiFile.

    `program_names` maps program numbers to human-readable track names
    (see note_event2midi).

    `beat_grid` is a detected beat grid (see muscriptor.utils.beats): it supplies
    the tempo, the time signature and a delay that puts bar lines on real
    downbeats. Defaults to PLACEHOLDER_GRID.

    The notes are moved onto the beat grid based on its `onset_delay` to better match
    the grid.

    `options` corrects the grid and says how far to tidy the notes onto it; see
    ExportOptions. Quantizing needs a subdivision, which `resolve_grid` may not
    find; the notes are then written unquantized.
    """
    options = options or ExportOptions()
    beat_grid = resolve_grid(beat_grid, options, [n.onset for n in notes])
    delay = beat_grid.onset_delay
    offset = beat_grid.bar_offset(min_shift=delay)
    notes = shifted_notes(notes, -delay)
    beat = 60.0 / beat_grid.bpm

    if options.quantize:
        if beat_grid.beat_subdivision is None:
            logger.warning("not quantizing: no beat subdivision to snap the notes to")
        else:
            step = beat / beat_grid.beat_subdivision
            notes = quantized_notes(notes, step, offset, options.strength)

    if options.fill_gaps > 0:
        notes = gaps_filled(notes, options.fill_gaps * beat)

    if options.trim_leading_bars and notes:
        offset -= leading_bars(notes, offset, beat_grid)

    return note_event2midi(
        note2note_event(notes),
        output_file=None,
        velocity=velocity,
        tempo=round(60_000_000 / beat_grid.bpm),
        program_names=program_names,
        beats_per_bar=beat_grid.beats_per_bar,
        offset_s=offset,
    )


def quantized_notes(
    notes: list[Note], step: float, offset_s: float, strength: float = 1.0
) -> list[Note]:
    """`notes` with every onset and offset moved onto a `step`-second grid.

    Snapped in the timeline the MIDI file will have — `offset_s` is the shift
    that puts bar 1 at tick 0 — so the notes land on whole grid steps from the
    start of the score rather than a constant fraction off it.

    `strength` below 1 only moves them that fraction of the way, which keeps
    some of the feel of the performance.
    """

    def snap(t: float) -> float:
        return round((t + offset_s) / step) * step - offset_s

    def pull(t: float, target: float) -> float:
        return t + strength * (target - t)

    snapped = []
    for note in notes:
        onset = snap(note.onset)
        # A note shorter than half a step rounds to nothing and would vanish
        # from the score; give it the shortest length the grid has.
        offset = max(snap(note.offset), onset + step)
        snapped.append(
            dataclasses.replace(
                note, onset=pull(note.onset, onset), offset=pull(note.offset, offset)
            )
        )
    # Bumping the short ones can make two notes of the same pitch overlap.
    return trim_overlapping_notes(snapped)


# Leeway for comparing times that went through the same float arithmetic.
_EPS = 1e-6


def gaps_filled(notes: list[Note], max_gap: float) -> list[Note]:
    """`notes` held on until their instrument's next onset, across short silences.

    A transcription ends notes where the sound fades, which notation software
    writes as a rest before the next note: a staccato score where the player
    let the notes ring. Per instrument, a note that stops at most `max_gap`
    seconds before the next onset (of any pitch) is extended to that onset.
    Drums have no length to extend.
    """
    by_instrument: dict[tuple[int, bool], list[Note]] = defaultdict(list)
    for note in notes:
        by_instrument[(note.program, note.is_drum)].append(note)

    filled = []
    for (_, is_drum), group in by_instrument.items():
        if is_drum:
            filled.extend(group)
            continue
        onsets = sorted({n.onset for n in group})
        for note in group:
            i = bisect.bisect_right(onsets, note.onset + _EPS)
            if i < len(onsets) and 0 < onsets[i] - note.offset <= max_gap + _EPS:
                note = dataclasses.replace(note, offset=onsets[i])
            filled.append(note)
    return filled


def leading_bars(notes: list[Note], offset_s: float, grid: BeatGrid) -> float:
    """Seconds of whole empty bars before the first note, in the MIDI timeline.

    Counted in beats when there is no meter, since that is all the grid has.
    """
    bar = grid.bar_seconds or 60.0 / grid.bpm
    first = min(n.onset for n in notes) + offset_s
    return max(0, math.floor((first + _EPS) / bar)) * bar
