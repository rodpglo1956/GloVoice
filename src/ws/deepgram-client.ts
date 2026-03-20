/**
 * Deepgram Nova-3 streaming STT client
 * Uses raw WebSocket to Deepgram's live transcription API (no SDK dependency).
 *
 * End-of-speech detection uses dual-trigger pattern (Deepgram recommended):
 *   1. speech_final: true — primary (endpointing VAD detected silence)
 *   2. UtteranceEnd — fallback (speech_final has known bugs with smart_format)
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
    no_delay: "true",
    interim_results: "true",
    utterance_end_ms: "1200",
    endpointing: "600",
    vad_events: "true",
  });

  const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;

  let ws: WebSocket | null = null;
  let open = false;

  // Track last transcript for UtteranceEnd fallback
  let lastTranscript = "";
  let speechFinalFired = false;

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

      // UtteranceEnd — fallback trigger when speech_final never fires
      // (known Deepgram bug with smart_format, see Discussion #409/#747)
      if (data.type === "UtteranceEnd") {
        if (!speechFinalFired && lastTranscript) {
          console.log("[Deepgram] UtteranceEnd fallback — speech_final never fired");
          onTranscript(lastTranscript, true);
          lastTranscript = "";
        }
        speechFinalFired = false;
        return;
      }

      const transcript = data.channel?.alternatives?.[0]?.transcript ?? "";
      if (!transcript) return;

      // speech_final: user stopped talking (endpointing VAD detected silence)
      if (data.speech_final === true) {
        speechFinalFired = true;
        lastTranscript = "";
        onTranscript(transcript, true);
        return;
      }

      // is_final: segment done processing (NOT end of speech — fires mid-sentence)
      // Store as last transcript for UtteranceEnd fallback, but don't trigger pipeline
      if (data.is_final === true) {
        lastTranscript = transcript;
        onTranscript(transcript, false); // send as interim for UI display
        return;
      }

      // Interim result — display only
      onTranscript(transcript, false);
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
        ws.send(JSON.stringify({ type: "KeepAlive" }));
      }
    },

    close() {
      if (ws) {
        open = false;
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
