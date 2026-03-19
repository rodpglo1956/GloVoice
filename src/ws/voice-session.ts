/**
 * VoiceSession — Per-connection state machine for streaming voice pipeline.
 * Manages Deepgram STT connection, audio forwarding, LLM-to-TTS pipeline,
 * and transcript delivery.
 */

import { createDeepgramConnection, type DeepgramConnection } from "./deepgram-client";
import {
  streamLLMToTTS,
  type LeadState,
  type StreamLLMToTTSResult,
} from "../pipeline/audio-pipeline";
import type OpenAI from "openai";

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

/** Config passed from server.ts at construction time */
export interface VoiceSessionConfig {
  openaiClient: OpenAI;
  llmModel: string;
  elevenLabsVoiceId: string;
  elevenLabsApiKey: string;
  buildPrompt: (industry: string) => string;
  onLeadCaptured?: (lead: LeadState, industry: string) => void;
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
  private lead: LeadState = { name: "", phone: "", email: "" };
  private config: VoiceSessionConfig;

  constructor(
    browserWs: { send(data: string | ArrayBuffer): number },
    config: VoiceSessionConfig
  ) {
    this.browserWs = browserWs;
    this.config = config;
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
   * Mutes during 'speaking' and 'processing' states to prevent echo.
   */
  onAudioFrame(data: ArrayBuffer): void {
    if (this.state === "listening" && this.deepgram) {
      this.deepgram.send(data);
    }
    // During 'speaking'/'processing' state: audio is intentionally dropped
    // to prevent TTS playback from being transcribed back (echo hallucination).
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
    this.lead = { name: "", phone: "", email: "" };
    console.log("[VoiceSession] Cleaned up");
  }

  /**
   * Get current session state.
   */
  getState(): SessionState {
    return this.state;
  }

  /**
   * Get conversation history.
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
  // Private: Listening / STT
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
        // Send transcript to browser for display
        this.sendToBrowser({
          type: "transcript",
          text,
          final: isFinal,
        });

        if (isFinal && text.trim()) {
          this.addToHistory("user", text.trim());
          // Trigger the LLM → TTS streaming pipeline
          this.handleFinalTranscript(text.trim());
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

  // -------------------------------------------------------------------------
  // Private: LLM → TTS Pipeline
  // -------------------------------------------------------------------------

  /**
   * Called when Deepgram returns a final transcript.
   * Triggers the streaming pipeline: LLM → TokenBuffer → ElevenLabs → browser.
   */
  private async handleFinalTranscript(userText: string): Promise<void> {
    // Transition to processing
    this.setState("processing");

    const systemPrompt = this.config.buildPrompt(this.industry);
    let firstAudioReceived = false;

    try {
      const result: StreamLLMToTTSResult = await streamLLMToTTS({
        userText,
        systemPrompt,
        history: this.conversationHistory,
        lead: this.lead,
        openaiClient: this.config.openaiClient,
        llmModel: this.config.llmModel,
        voiceId: this.config.elevenLabsVoiceId,
        elevenLabsApiKey: this.config.elevenLabsApiKey,

        // onAudio: send binary PCM to browser, set speaking state on first chunk
        onAudio: (pcmBuffer: ArrayBuffer) => {
          if (!firstAudioReceived) {
            firstAudioReceived = true;
            this.setSpeaking();
          }
          this.sendBinaryToBrowser(pcmBuffer);
        },

        // onSpoken: add assistant response to conversation history
        onSpoken: (fullText: string) => {
          this.addToHistory("assistant", fullText);
        },

        // onDone: all audio finished
        onDone: () => {
          // Handled after the await below
        },
      });

      // Handle lead capture
      if (result.save_lead) {
        // Merge any lead data from the LLM response
        if (result.lead.name) this.lead.name = result.lead.name;
        if (result.lead.phone) this.lead.phone = result.lead.phone;
        if (result.lead.email) this.lead.email = result.lead.email;

        // Notify server for Supabase save + email + Telegram
        this.config.onLeadCaptured?.(this.lead, this.industry);
      }

      // Handle end call
      if (result.end_call) {
        this.sendToBrowser({ type: "end_call" });
        return; // Don't resume listening
      }

      // Resume listening for next user utterance
      this.resumeListening();
    } catch (err) {
      console.error("[VoiceSession] Pipeline error:", err);
      this.sendToBrowser({
        type: "error",
        message: "Processing error",
      });
      this.resumeListening();
    }
  }

  // -------------------------------------------------------------------------
  // Private: State + Messaging
  // -------------------------------------------------------------------------

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

  private sendBinaryToBrowser(buffer: ArrayBuffer): void {
    try {
      this.browserWs.send(buffer);
    } catch (err) {
      console.error("[VoiceSession] Failed to send binary to browser:", err);
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
