/**
 * VAD Processor — Server-side Voice Activity Detection for barge-in.
 * Uses energy-based detection (RMS threshold) instead of ONNX/Silero
 * to avoid native binary compatibility issues on Railway/Bun.
 *
 * For barge-in, we only need to detect "is the user talking while Marie
 * is speaking?" — energy-based detection is sufficient for this.
 *
 * Tuned to reject speaker echo: browser echo cancellation helps but
 * isn't perfect on laptops. High RMS threshold + many consecutive frames
 * ensures only deliberate speech triggers barge-in.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VADProcessor {
  /**
   * Process a PCM audio frame. Returns true if speech is detected.
   * Input: Int16 PCM ArrayBuffer at 16kHz mono.
   */
  processFrame(pcmBuffer: ArrayBuffer): boolean;
  /** Reset VAD state (e.g., after a barge-in is handled). */
  reset(): void;
  /** Start a cooldown period where VAD ignores all input (e.g., when Marie starts speaking). */
  startCooldown(ms: number): void;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const RMS_THRESHOLD = 2000; // Int16 RMS — raised from 800 to reject speaker echo
const SPEECH_FRAMES_REQUIRED = 6; // Consecutive frames above threshold — raised from 3
const SILENCE_FRAMES_REQUIRED = 5; // Consecutive frames below threshold to clear

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Create an energy-based VAD processor.
 * No external dependencies, works on any runtime.
 */
export function createVADProcessor(): VADProcessor {
  let speechFrameCount = 0;
  let silenceFrameCount = 0;
  let isSpeaking = false;
  let cooldownUntil = 0;

  return {
    processFrame(pcmBuffer: ArrayBuffer): boolean {
      // During cooldown, ignore all audio (prevents echo from triggering barge-in)
      if (Date.now() < cooldownUntil) {
        return false;
      }

      const int16 = new Int16Array(pcmBuffer);

      // Calculate RMS energy
      let sumSquares = 0;
      for (let i = 0; i < int16.length; i++) {
        sumSquares += int16[i] * int16[i];
      }
      const rms = Math.sqrt(sumSquares / int16.length);

      if (rms > RMS_THRESHOLD) {
        speechFrameCount++;
        silenceFrameCount = 0;

        if (speechFrameCount >= SPEECH_FRAMES_REQUIRED) {
          isSpeaking = true;
        }
      } else {
        silenceFrameCount++;
        speechFrameCount = 0;

        if (silenceFrameCount >= SILENCE_FRAMES_REQUIRED) {
          isSpeaking = false;
        }
      }

      return isSpeaking;
    },

    reset(): void {
      speechFrameCount = 0;
      silenceFrameCount = 0;
      isSpeaking = false;
    },

    startCooldown(ms: number): void {
      cooldownUntil = Date.now() + ms;
      speechFrameCount = 0;
      silenceFrameCount = 0;
      isSpeaking = false;
    },
  };
}
