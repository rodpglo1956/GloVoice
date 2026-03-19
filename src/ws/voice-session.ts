/**
 * VoiceSession — Per-connection state machine for streaming voice pipeline.
 * Manages Deepgram STT connection, audio forwarding, and transcript delivery.
 */

import { createDeepgramConnection, type DeepgramConnection } from "./deepgram-client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SessionState = "idle" | "listening" | "processing" | "speaking" | "interrupted";

interface ControlMessage {
  type: string;
  industry?: string;
  [key: string]: unknown;
}

interface ConversationEntry {
  role: "user" | "assistant";
  content: string;
}

// ---------------------------------------------------------------------------
// VoiceSession
// ---------------------------------------------------------------------------

export class VoiceSession {
  private browserWs: { send(data: string | ArrayBuffer): number };
  private state: SessionState = "idle";
  private deepgram: DeepgramConnection | null = null;
  private industry: string = "other";
  private conversationHistory: ConversationEntry[] = [];
  private keepAliveInterval: ReturnType<typeof setInterval> | null = null;

  constructor(browserWs: { send(data: string | ArrayBuffer): number }) {
    this.browserWs = browserWs;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Handle JSON control messages from the browser.
   */
  handleControl(data: ControlMessage): void {
    switch (data.type) {
      case "start":
        this.startListening(data.industry || "other");
        break;
      case "end":
        this.cleanup();
        break;
      default:
        console.warn(`[VoiceSession] Unknown control type: ${data.type}`);
    }
  }

  /**
   * Handle binary audio frames from the browser.
   * Only forwards to Deepgram when in 'listening' state.
   * Mutes during 'speaking' state to prevent echo hallucination.
   */
  onAudioFrame(data: ArrayBuffer): void {
    if (this.state === "listening" && this.deepgram) {
      this.deepgram.send(data);
    }
    // During 'speaking' state: audio is intentionally dropped to prevent
    // TTS playback from being transcribed back (echo hallucination).
    // Future: VAD stub -- could detect barge-in during speaking state.
  }

  /**
   * Clean up all resources for this session.
   */
  cleanup(): void {
    this.stopKeepAlive();

    if (this.deepgram) {
      this.deepgram.close();
      this.deepgram = null;
    }

    this.setState("idle");
    this.conversationHistory = [];
    console.log("[VoiceSession] Cleaned up");
  }

  /**
   * Get current session state.
   */
  getState(): SessionState {
    return this.state;
  }

  /**
   * Get conversation history (for future LLM integration).
   */
  getHistory(): ConversationEntry[] {
    return this.conversationHistory;
  }

  /**
   * Get industry for this session.
   */
  getIndustry(): string {
    return this.industry;
  }

  /**
   * Transition to speaking state (called when TTS starts playing).
   * Mutes Deepgram forwarding and starts keepalive.
   */
  setSpeaking(): void {
    this.setState("speaking");
  }

  /**
   * Transition back to listening after speaking completes.
   */
  resumeListening(): void {
    if (this.state === "speaking" || this.state === "processing") {
      this.setState("listening");
    }
  }

  /**
   * Add an entry to conversation history.
   */
  addToHistory(role: "user" | "assistant", content: string): void {
    this.conversationHistory.push({ role, content });
    // Cap history at 20 entries (10 turns)
    if (this.conversationHistory.length > 20) {
      this.conversationHistory = this.conversationHistory.slice(-20);
    }
  }

  // -------------------------------------------------------------------------
  // Private
  // -------------------------------------------------------------------------

  private startListening(industry: string): void {
    this.industry = industry;

    // Close existing Deepgram connection if any
    if (this.deepgram) {
      this.deepgram.close();
      this.deepgram = null;
    }

    this.deepgram = createDeepgramConnection(
      // onTranscript
      (text: string, isFinal: boolean) => {
        this.sendToBrowser({
          type: "transcript",
          text,
          final: isFinal,
        });

        if (isFinal && text.trim()) {
          this.addToHistory("user", text.trim());
        }
      },
      // onError
      (error: Error) => {
        console.error("[VoiceSession] Deepgram error:", error.message);
        this.sendToBrowser({
          type: "error",
          message: "Speech recognition error",
        });
      },
      // onClose
      () => {
        console.log("[VoiceSession] Deepgram connection closed");
      }
    );

    this.setState("listening");
    this.startKeepAlive();

    console.log(`[VoiceSession] Started listening for industry: ${industry}`);
  }

  private setState(newState: SessionState): void {
    const oldState = this.state;
    this.state = newState;

    if (oldState !== newState) {
      this.sendToBrowser({
        type: "status",
        state: newState,
      });
    }
  }

  private sendToBrowser(data: Record<string, unknown>): void {
    try {
      this.browserWs.send(JSON.stringify(data));
    } catch (err) {
      console.error("[VoiceSession] Failed to send to browser:", err);
    }
  }

  /**
   * Keep Deepgram connection alive during non-audio states (processing, speaking).
   * Sends keepalive every 5 seconds when not actively forwarding audio.
   */
  private startKeepAlive(): void {
    this.stopKeepAlive();
    this.keepAliveInterval = setInterval(() => {
      if (this.deepgram && this.state !== "listening") {
        this.deepgram.keepAlive();
      }
    }, 5000);
  }

  private stopKeepAlive(): void {
    if (this.keepAliveInterval) {
      clearInterval(this.keepAliveInterval);
      this.keepAliveInterval = null;
    }
  }
}
