/**
 * ElevenLabs WebSocket TTS client — streams text in, receives PCM audio out.
 *
 * Protocol: wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input
 *   BOS  → { text: ' ', voice_settings, generation_config, xi_api_key }
 *   Text → { text: 'chunk', flush: false }
 *   Flush→ { text: 'end of sentence. ', flush: true }
 *   EOS  → { text: '' }
 *   Audio← { audio: 'base64_pcm_data', isFinal: false }
 */

export interface ElevenLabsWSClient {
  /** Send a text chunk. If flush=true, ElevenLabs generates audio immediately. */
  sendText(text: string, flush?: boolean): void;
  /** Send end-of-stream signal (EOS). */
  endStream(): void;
  /** Force close the connection. */
  close(): void;
}

interface ElevenLabsConfig {
  voiceId: string;
  apiKey: string;
}

/**
 * Open a WebSocket connection to ElevenLabs streaming TTS.
 *
 * @param config  - voiceId and apiKey for the connection
 * @param onAudio - Called with base64-encoded PCM audio chunks
 * @param onDone  - Called when ElevenLabs signals generation is complete
 */
export function createElevenLabsWS(
  config: ElevenLabsConfig,
  onAudio: (pcmBase64: string) => void,
  onDone: () => void
): ElevenLabsWSClient {
  const url =
    `wss://api.elevenlabs.io/v1/text-to-speech/${config.voiceId}/stream-input` +
    `?model_id=eleven_flash_v2_5&output_format=pcm_16000`;

  let ws: WebSocket | null = null;
  let open = false;
  let eosQueued = false;

  try {
    ws = new WebSocket(url);
  } catch (err) {
    console.error("[ElevenLabs] Failed to create WebSocket:", err);
    setTimeout(onDone, 0);
    return { sendText() {}, endStream() {}, close() {} };
  }

  ws.addEventListener("open", () => {
    open = true;
    // Send Beginning-of-Stream (BOS) message
    const bos = {
      text: " ",
      voice_settings: {
        stability: 0.70,
        similarity_boost: 0.75,
        use_speaker_boost: true,
      },
      generation_config: {
        chunk_length_schedule: [50, 90, 120, 150],
      },
      xi_api_key: config.apiKey,
    };
    ws!.send(JSON.stringify(bos));
    console.log("[ElevenLabs] WebSocket opened, BOS sent");
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    try {
      const data = typeof event.data === "string" ? JSON.parse(event.data) : {};
      if (data.audio) {
        onAudio(data.audio);
      }
      if (data.isFinal) {
        onDone();
      }
    } catch (err) {
      console.error("[ElevenLabs] Failed to parse message:", err);
    }
  });

  ws.addEventListener("error", (event: Event) => {
    console.error("[ElevenLabs] WebSocket error:", event);
    open = false;
    onDone();
  });

  ws.addEventListener("close", () => {
    open = false;
    console.log("[ElevenLabs] WebSocket closed");
  });

  return {
    sendText(text: string, flush = false) {
      if (ws && open) {
        ws.send(JSON.stringify({ text, flush }));
      }
    },

    endStream() {
      if (ws && open && !eosQueued) {
        eosQueued = true;
        // EOS: empty text signals end of input
        ws.send(JSON.stringify({ text: "" }));
        console.log("[ElevenLabs] EOS sent");
      }
    },

    close() {
      open = false;
      if (ws) {
        try {
          ws.close();
        } catch {
          // ignore
        }
        ws = null;
      }
    },
  };
}
