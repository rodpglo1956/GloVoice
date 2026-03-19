/**
 * VAD Processor — Server-side Voice Activity Detection for barge-in.
 * Uses avr-vad (ONNX-based Silero VAD) to detect when the user speaks
 * while the agent is playing TTS audio (barge-in detection).
 */

import { RealTimeVAD } from "avr-vad";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VADProcessor {
  /**
   * Process a PCM audio frame. Returns true if speech is detected.
   * Input: Int16 PCM ArrayBuffer at 16kHz mono.
   */
  processFrame(pcmBuffer: ArrayBuffer): Promise<boolean>;
  /** Reset VAD state (e.g., after a barge-in is handled). */
  reset(): void;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Create a VAD processor backed by avr-vad (Silero VAD v5).
 *
 * Returns null if initialization fails (caller should handle gracefully).
 */
export async function createVADProcessor(): Promise<VADProcessor | null> {
  try {
    const vad = await RealTimeVAD.new({
      model: "v5",
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      frameSamples: 1536,
    });

    return {
      async processFrame(pcmBuffer: ArrayBuffer): Promise<boolean> {
        // Convert Int16 PCM to Float32 [-1, 1] for VAD
        const int16 = new Int16Array(pcmBuffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
          float32[i] = int16[i] / 32768;
        }

        try {
          const result = await vad.processFrame(float32);
          return (
            result.msg === "SPEECH_START" || result.msg === "SPEECH_CONTINUE"
          );
        } catch {
          return false;
        }
      },

      reset(): void {
        try {
          vad.reset();
        } catch {
          // ignore reset errors
        }
      },
    };
  } catch (err) {
    console.error("[VAD] Failed to initialize avr-vad:", err);
    return null;
  }
}
