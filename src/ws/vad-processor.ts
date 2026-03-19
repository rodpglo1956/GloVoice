/**
 * VAD Processor — Server-side Voice Activity Detection for barge-in.
 * Uses energy-based detection (RMS threshold) instead of ONNX/Silero
 * to avoid native binary compatibility issues on Railway/Bun.
 *
 * For barge-in, we only need to detect "is the user talking while Marie
 * is speaking?" — energy-based detection is sufficient for this.
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
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const RMS_THRESHOLD = 800; // Int16 RMS threshold for "speech" (tune as needed)
const SPEECH_FRAMES_REQUIRED = 3; // Consecutive frames above threshold to trigger
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

  return {
    processFrame(pcmBuffer: ArrayBuffer): boolean {
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
  };
}
