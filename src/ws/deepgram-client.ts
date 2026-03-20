/**
 * Deepgram Nova-3 streaming STT client
 * Uses raw WebSocket to Deepgram's live transcription API (no SDK dependency).
 */

const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY || "";

export interface DeepgramConnection {
  send(pcmBuffer: ArrayBuffer | Buffer): void;
  keepAlive(): void;
  close(): void;
  isOpen(): boolean;
}

interface DeepgramTranscriptResult {
  channel?: {
    alternatives?: Array<{
      transcript?: string;
    }>;
  };
  speech_final?: boolean;
  is_final?: boolean;
  type?: string;
}

/**
 * Create a live Deepgram connection for streaming STT.
 *
 * @param onTranscript - Called with (text, isFinal) when Deepgram returns a transcript
 * @param onError - Called on connection errors
 * @param onClose - Called when the connection closes
 */
export function createDeepgramConnection(
  onTranscript: (text: string, isFinal: boolean) => void,
  onError?: (error: Error) => void,
  onClose?: () => void
): DeepgramConnection {
  if (!DEEPGRAM_API_KEY) {
    console.error("[Deepgram] DEEPGRAM_API_KEY not set");
    return {
      send() {},
      keepAlive() {},
      close() {},
      isOpen() { return false; },
    };
  }

  const params = new URLSearchParams({
    model: "nova-3",
    language: "en-US",
    encoding: "linear16",
    sample_rate: "16000",
    channels: "1",
    smart_format: "true",
    interim_results: "true",
    utterance_end_ms: "1500",
    endpointing: "800",
    vad_events: "true",
  });

  const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;

  let ws: WebSocket | null = null;
  let open = false;

  try {
    ws = new WebSocket(url, {
      headers: {
        Authorization: `Token ${DEEPGRAM_API_KEY}`,
      },
    } as any);
  } catch (err) {
    console.error("[Deepgram] Failed to create WebSocket:", err);
    onError?.(err instanceof Error ? err : new Error(String(err)));
    return {
      send() {},
      keepAlive() {},
      close() {},
      isOpen() { return false; },
    };
  }

  ws.addEventListener("open", () => {
    open = true;
    console.log("[Deepgram] Connection opened");
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    try {
      const data: DeepgramTranscriptResult =
        typeof event.data === "string" ? JSON.parse(event.data) : {};

      // VAD events and metadata don't have transcripts
      if (data.type === "UtteranceEnd") {
        // Utterance boundary -- treat as final empty if needed
        return;
      }

      const transcript = data.channel?.alternatives?.[0]?.transcript ?? "";
      if (!transcript) return;

      // ONLY use speech_final — is_final fires on every segment boundary (mid-sentence).
      // speech_final fires when Deepgram detects the user actually stopped talking.
      const isFinal = data.speech_final === true;
      onTranscript(transcript, isFinal);
    } catch (err) {
      console.error("[Deepgram] Failed to parse message:", err);
    }
  });

  ws.addEventListener("error", (event: Event) => {
    console.error("[Deepgram] WebSocket error:", event);
    onError?.(new Error("Deepgram WebSocket error"));
  });

  ws.addEventListener("close", () => {
    open = false;
    console.log("[Deepgram] Connection closed");
    onClose?.();
  });

  return {
    send(pcmBuffer: ArrayBuffer | Buffer) {
      if (ws && open) {
        ws.send(pcmBuffer);
      }
    },

    keepAlive() {
      if (ws && open) {
        // Deepgram keepalive: send a JSON keepalive message
        ws.send(JSON.stringify({ type: "KeepAlive" }));
      }
    },

    close() {
      if (ws) {
        open = false;
        // Send CloseStream message for graceful shutdown
        try {
          ws.send(JSON.stringify({ type: "CloseStream" }));
        } catch {
          // ignore if already closed
        }
        ws.close();
        ws = null;
      }
    },

    isOpen() {
      return open;
    },
  };
}
