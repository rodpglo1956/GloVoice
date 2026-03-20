/**
 * ElevenLabs WebSocket TTS client — streams text in, receives PCM audio out.
 *
 * Two modes:
 * 1. Per-turn: createElevenLabsWS() — opens a new WS per turn (legacy)
 * 2. Persistent: createPersistentElevenLabsWS() — reuses one WS across turns,
 *    sending BOS/text/EOS cycles on the same connection (saves 150-300ms handshake)
 */

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

export interface ElevenLabsWSClient {
  sendText(text: string, flush?: boolean): void;
  endStream(): void;
  close(): void;
}

export interface ElevenLabsConfig {
  voiceId: string;
  apiKey: string;
}

export interface ElevenLabsContext {
  sendText(text: string, flush?: boolean): void;
  endStream(): void;
  close(): void;
}

export interface PersistentElevenLabsWS {
  createContext(
    onAudio: (pcmBase64: string) => void,
    onDone: () => void
  ): ElevenLabsContext;
  destroy(): void;
  isConnected(): boolean;
}

// ---------------------------------------------------------------------------
// Voice settings (shared)
// ---------------------------------------------------------------------------

const VOICE_SETTINGS = {
  stability: 0.70,
  similarity_boost: 0.75,
  use_speaker_boost: true,
};

const GENERATION_CONFIG = {
  chunk_length_schedule: [50, 90, 120, 150],
};

// ---------------------------------------------------------------------------
// Per-turn client (legacy, kept for fallback)
// ---------------------------------------------------------------------------

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
  let closed = false;
  let eosQueued = false;
  let doneFired = false;
  const pendingQueue: string[] = [];

  function fireDone() {
    if (!doneFired) {
      doneFired = true;
      onDone();
    }
  }

  try {
    ws = new WebSocket(url);
  } catch (err) {
    console.error("[ElevenLabs] Failed to create WebSocket:", err);
    setTimeout(fireDone, 0);
    return { sendText() {}, endStream() {}, close() {} };
  }

  ws.addEventListener("open", () => {
    open = true;
    const bos = {
      text: " ",
      voice_settings: VOICE_SETTINGS,
      generation_config: GENERATION_CONFIG,
      xi_api_key: config.apiKey,
    };
    ws!.send(JSON.stringify(bos));

    for (const msg of pendingQueue) {
      ws!.send(msg);
    }
    pendingQueue.length = 0;
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    try {
      const data = typeof event.data === "string" ? JSON.parse(event.data) : {};
      if (data.audio) onAudio(data.audio);
      if (data.isFinal) fireDone();
    } catch (err) {
      console.error("[ElevenLabs] Failed to parse message:", err);
    }
  });

  ws.addEventListener("error", () => {
    open = false;
    fireDone();
  });

  ws.addEventListener("close", () => {
    open = false;
    fireDone();
  });

  return {
    sendText(text: string, flush = false) {
      if (closed) return;
      const msg = JSON.stringify({ text, flush });
      if (ws && open) ws.send(msg);
      else if (ws) pendingQueue.push(msg);
    },
    endStream() {
      if (closed || eosQueued) return;
      eosQueued = true;
      const msg = JSON.stringify({ text: "" });
      if (ws && open) ws.send(msg);
      else if (ws) pendingQueue.push(msg);
    },
    close() {
      closed = true;
      open = false;
      pendingQueue.length = 0;
      if (ws) { try { ws.close(); } catch {} ws = null; }
    },
  };
}

// ---------------------------------------------------------------------------
// Persistent client (reuses one WS across turns)
// ---------------------------------------------------------------------------

/**
 * Open a persistent WebSocket to ElevenLabs. Each turn creates a "context"
 * by sending a fresh BOS, text chunks, and EOS on the same connection.
 * Saves 150-300ms TCP+TLS handshake per turn.
 */
export function createPersistentElevenLabsWS(
  config: ElevenLabsConfig
): PersistentElevenLabsWS {
  const url =
    `wss://api.elevenlabs.io/v1/text-to-speech/${config.voiceId}/stream-input` +
    `?model_id=eleven_flash_v2_5&output_format=pcm_16000`;

  let ws: WebSocket | null = null;
  let open = false;
  let destroyed = false;

  // Active context callbacks
  let activeOnAudio: ((pcmBase64: string) => void) | null = null;
  let activeOnDone: (() => void) | null = null;
  let activeDoneFired = false;

  function fireActiveDone() {
    if (!activeDoneFired && activeOnDone) {
      activeDoneFired = true;
      activeOnDone();
    }
  }

  try {
    ws = new WebSocket(url);
  } catch (err) {
    console.error("[ElevenLabs] Failed to create persistent WebSocket:", err);
    return {
      createContext(onAudio, onDone) {
        setTimeout(onDone, 0);
        return { sendText() {}, endStream() {}, close() {} };
      },
      destroy() {},
      isConnected() { return false; },
    };
  }

  ws.addEventListener("open", () => {
    open = true;
    console.log("[ElevenLabs] Persistent WS opened");
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    try {
      const data = typeof event.data === "string" ? JSON.parse(event.data) : {};
      if (data.audio && activeOnAudio) {
        activeOnAudio(data.audio);
      }
      if (data.isFinal) {
        fireActiveDone();
      }
    } catch (err) {
      console.error("[ElevenLabs] Failed to parse message:", err);
    }
  });

  ws.addEventListener("error", () => {
    open = false;
    fireActiveDone();
  });

  ws.addEventListener("close", () => {
    open = false;
    console.log("[ElevenLabs] Persistent WS closed");
    fireActiveDone();
  });

  return {
    createContext(
      onAudio: (pcmBase64: string) => void,
      onDone: () => void
    ): ElevenLabsContext {
      // Set active context callbacks
      activeOnAudio = onAudio;
      activeOnDone = onDone;
      activeDoneFired = false;

      let contextClosed = false;
      let eosQueued = false;
      const pendingQueue: string[] = [];

      // Send BOS for this turn
      const bos = {
        text: " ",
        voice_settings: VOICE_SETTINGS,
        generation_config: GENERATION_CONFIG,
        xi_api_key: config.apiKey,
      };

      if (ws && open) {
        ws.send(JSON.stringify(bos));
      } else {
        // WS not open yet — queue BOS and everything after it
        pendingQueue.push(JSON.stringify(bos));
        // Set up a one-time flush when WS opens
        const flushOnOpen = () => {
          if (ws && open && pendingQueue.length > 0) {
            for (const msg of pendingQueue) {
              ws.send(msg);
            }
            pendingQueue.length = 0;
          }
        };
        ws?.addEventListener("open", flushOnOpen, { once: true });
      }

      return {
        sendText(text: string, flush = false) {
          if (contextClosed || destroyed) return;
          const msg = JSON.stringify({ text, flush });
          if (ws && open) ws.send(msg);
          else pendingQueue.push(msg);
        },
        endStream() {
          if (contextClosed || destroyed || eosQueued) return;
          eosQueued = true;
          const msg = JSON.stringify({ text: "" });
          if (ws && open) ws.send(msg);
          else pendingQueue.push(msg);
        },
        close() {
          // Close just this context's stream, not the persistent connection
          contextClosed = true;
          if (!eosQueued && ws && open) {
            ws.send(JSON.stringify({ text: "" }));
          }
          fireActiveDone();
          activeOnAudio = null;
          activeOnDone = null;
        },
      };
    },

    destroy() {
      destroyed = true;
      open = false;
      activeOnAudio = null;
      activeOnDone = null;
      if (ws) {
        try { ws.close(); } catch {}
        ws = null;
      }
      console.log("[ElevenLabs] Persistent WS destroyed");
    },

    isConnected() {
      return open;
    },
  };
}
