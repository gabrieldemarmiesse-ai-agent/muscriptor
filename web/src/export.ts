import type { BeatGrid } from "./pianoroll";

/** A note as /transcribe streamed it — times not yet moved by `onset_delay`. */
export type ExportNote = {
  instrument: string;
  pitch: number;
  start: number;
  end: number;
};

/** The user's corrections to the detected grid; null keeps what was detected. */
export type TimingOverrides = {
  bpm: number | null;
  beats_per_bar: number | null;
  /** A bar line, in seconds of the original audio. */
  first_downbeat: number | null;
};

export const NO_OVERRIDES: TimingOverrides = {
  bpm: null,
  beats_per_bar: null,
  first_downbeat: null,
};

/** How the notes are tidied up on export (the backend's ExportOptions). */
export type NoteCleanup = {
  quantize: boolean;
  /** Steps per beat to snap to; null for the one the onsets were measured on. */
  subdivision: number | null;
  /** 0..1, how far to move notes onto the grid. */
  strength: number;
  /** Hold notes across silences up to this many beats; 0 disables. */
  fill_gaps: number;
  trim_leading_bars: boolean;
};

export const DEFAULT_CLEANUP: NoteCleanup = {
  quantize: true,
  subdivision: null,
  strength: 1,
  fill_gaps: 0.25,
  trim_leading_bars: false,
};

/**
 * The grid the bar lines are drawn on: the detected one with the user's
 * corrections, or one made up from scratch when the user typed a tempo for a
 * recording where none was detected. null while there is no tempo at all.
 */
export function correctedGrid(
  detected: BeatGrid | null,
  timing: TimingOverrides,
): BeatGrid | null {
  const bpm = timing.bpm ?? detected?.bpm;
  if (bpm === undefined) return null;
  return {
    bpm,
    beats_per_bar: timing.beats_per_bar ?? detected?.beats_per_bar ?? null,
    first_downbeat: timing.first_downbeat ?? detected?.first_downbeat ?? 0,
    onset_delay: detected?.onset_delay ?? 0,
    beat_subdivision: detected?.beat_subdivision ?? null,
  };
}

/** Whether the user changed the tempo, which invalidates the detected subdivision. */
export function isRetimed(detected: BeatGrid | null, timing: TimingOverrides): boolean {
  return timing.bpm !== null && timing.bpm !== detected?.bpm;
}

/** The .mid for `notes`, written by the server's /export with these options. */
export async function exportMidi(
  notes: ExportNote[],
  detected: BeatGrid | null,
  timing: TimingOverrides,
  cleanup: NoteCleanup,
): Promise<Blob> {
  const resp = await fetch("/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      notes,
      beat_grid: detected && {
        bpm: detected.bpm,
        beats_per_bar: detected.beats_per_bar,
        first_downbeat: detected.first_downbeat,
        onset_delay: detected.onset_delay,
        beat_subdivision: detected.beat_subdivision ?? null,
      },
      options: { ...timing, ...cleanup },
    }),
  });
  if (!resp.ok) throw new Error(await errorDetail(resp));
  return new Blob([await resp.arrayBuffer()], { type: "audio/midi" });
}

/** The FastAPI `detail` of a failed response, or its raw body if it isn't one. */
export async function errorDetail(resp: Response): Promise<string> {
  const text = await resp.text();
  try {
    const detail = JSON.parse(text).detail ?? text;
    // Validation errors (422) come as a list of {loc, msg} objects.
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  } catch {
    // not JSON — keep the raw body
    return text;
  }
}

/** Save `blob` as `filename` through a throwaway object URL. */
export function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  // Revoked on the next tick: the click has started the download by then.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
