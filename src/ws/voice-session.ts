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
import { createVADProcessor, type VADProcessor } from "./vad-processor";
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
  private vadProcessor: VADProcessor | null = null;
  private consecutiveSpeechFrames = 0;
  private static readonly BARGE_IN_THRESHOLD = 3; // Require 3 consecutive speech frames
  /** Active ElevenLabs WS handle — stored so barge-in can close it */
  private activeElevenLabsClose: (() => void) | null = null;
  /** Timestamp of last speech_final / utterance_end from Deepgram */
  private speechEndTime = 0;
  /** Timestamp when first audio chunk sent to browser for current turn */
  private firstAudioSentTime = 0;
  /** Whether this is the greeting turn (no user input yet) */
  private isGreeting = false;
  /** Guard: true while handleFinalTranscript is running (prevents concurrent calls) */
  private pipelineRunning = false;

  constructor(
    browserWs: { send(data: string | ArrayBuffer): number },
    config: VoiceSessionConfig
  ) {
    this.browserWs = browserWs;
    this.config = config;
    // Initialize energy-based VAD (synchronous, no native deps)
    this.vadProcessor = createVADProcessor();
    console.log("[VoiceSession] VAD processor initialized (energy-based)");
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
   * During 'listening': forward to Deepgram for STT.
   * During 'speaking': route to VAD for barge-in detection AND keep Deepgram fed.
   */
  onAudioFrame(data: ArrayBuffer): void {
    if (this.state === "listening" && this.deepgram) {
      this.deepgram.send(data);
    } else if (this.state === "speaking") {
      // Keep Deepgram fed so it's ready when barge-in triggers
      if (this.deepgram) {
        this.deepgram.send(data);
      }
      // Run VAD for barge-in detection
      if (this.vadProcessor && this.state === "speaking") {
        const isSpeech = this.vadProcessor.processFrame(data);
        if (isSpeech) {
          this.consecutiveSpeechFrames++;
          if (this.consecutiveSpeechFrames >= VoiceSession.BARGE_IN_THRESHOLD) {
            this.handleBargeIn();
          }
        } else {
          this.consecutiveSpeechFrames = 0;
        }
      }
    }
    // During 'processing' state: still feed Deepgram so it stays ready
    else if (this.state === "processing" && this.deepgram) {
      this.deepgram.send(data);
    }
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

    if (this.activeElevenLabsClose) {
      this.activeElevenLabsClose();
      this.activeElevenLabsClose = null;
    }

    this.vadProcessor?.reset();
    this.consecutiveSpeechFrames = 0;
    this.pipelineRunning = false;
    this.setState("idle");
    this.conversationHistory = [];
    this.lead = { name: "", phone: "", email: "" };
    console.log("[VoiceSession] Cleaned up");
  }

  /**
   * Handle barge-in: user spoke while agent was speaking.
   * Stops TTS, clears browser playback, transitions to listening.
   */
  private handleBargeIn(): void {
    if (this.state !== "speaking") return;

    console.log("[VoiceSession] Barge-in detected — interrupting agent");

    // 1. Set state to interrupted (then quickly to listening)
    this.setState("interrupted");

    // 2. Close active ElevenLabs connection to stop TTS generation
    if (this.activeElevenLabsClose) {
      this.activeElevenLabsClose();
      this.activeElevenLabsClose = null;
    }

    // 3. Tell browser to clear its playback queue
    this.sendToBrowser({ type: "stop_playback" });

    // 4. Reset VAD state
    this.vadProcessor?.reset();
    this.consecutiveSpeechFrames = 0;

    // 5. Transition to listening — Deepgram is already receiving audio
    this.setState("listening");
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
   */
  setSpeaking(): void {
    this.setState("speaking");
  }

  /**
   * Transition back to listening after speaking completes.
   */
  resumeListening(): void {
    if (this.state === "speaking" || this.state === "processing" || this.state === "interrupted") {
      this.setState("listening");
    }
  }

  /**
   * Add an entry to conversation history.
   * For assistant entries, wrap in JSON so the LLM sees consistent format.
   */
  addToHistory(role: "user" | "assistant", content: string): void {
    if (role === "assistant") {
      // Store as JSON so the LLM sees the same format it's expected to output.
      // Prevents the model from drifting to plain text after seeing non-JSON history.
      const jsonContent = JSON.stringify({ spoken: content, save_lead: false, end_call: false });
      this.conversationHistory.push({ role, content: jsonContent });
    } else {
      this.conversationHistory.push({ role, content });
    }
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

    // Track whether we've already triggered for the current utterance
    let lastHandledTranscript = "";

    this.deepgram = createDeepgramConnection(
      // onTranscript
      (text: string, isFinal: boolean) => {
        if (isFinal) {
          // Record when user stopped speaking (for latency measurement)
          this.speechEndTime = Date.now();
        }

        // Send transcript to browser for display (include speechEndTime for client-side latency)
        this.sendToBrowser({
          type: "transcript",
          text,
          final: isFinal,
          ...(isFinal ? { speechEndTime: this.speechEndTime } : {}),
        });

        if (isFinal && text.trim()) {
          // Guard: skip if we already handled this exact transcript
          // (Deepgram can fire both is_final and speech_final for the same text)
          if (text.trim() === lastHandledTranscript) {
            console.log("[VoiceSession] Skipping duplicate final transcript");
            return;
          }
          lastHandledTranscript = text.trim();

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

    this.startKeepAlive();

    console.log(`[VoiceSession] Started listening for industry: ${industry}`);

    // Trigger greeting immediately
    this.playGreeting();
  }

  // -------------------------------------------------------------------------
  // Private: Greeting
  // -------------------------------------------------------------------------

  /**
   * Play an initial greeting through the streaming pipeline.
   * Uses the same LLM→TTS flow so the greeting is streamed with low latency.
   */
  private async playGreeting(): Promise<void> {
    this.isGreeting = true;

    const industryNames: Record<string, string> = {
      transportation: "Transportation",
      commercial: "Commercial Services",
      trades: "Trades",
      health: "Health & Wellness",
      realestate: "Real Estate",
      automotive: "Automotive",
      beauty: "Beauty",
      food: "Food & Hospitality",
      other: "Business Services",
    };
    const industryLabel = industryNames[this.industry] || industryNames.other;

    const greetingPrompt = `You are starting a new call. Greet the caller warmly. Say something like: "Hey, this is Marie with Glo Matrix. I see you're interested in our ${industryLabel} solutions. Tell me a little about your business and I'll show you what we can do." Keep it natural and under 2 sentences. Output valid JSON: {"spoken":"your greeting","save_lead":false,"end_call":false}`;

    const systemPrompt = this.config.buildPrompt(this.industry);
    let firstAudioReceived = false;

    try {
      await streamLLMToTTS({
        userText: greetingPrompt,
        systemPrompt,
        history: [],
        lead: this.lead,
        openaiClient: this.config.openaiClient,
        llmModel: this.config.llmModel,
        voiceId: this.config.elevenLabsVoiceId,
        elevenLabsApiKey: this.config.elevenLabsApiKey,

        onTTSReady: (closeFn: () => void) => {
          this.activeElevenLabsClose = closeFn;
        },

        onAudio: (pcmBuffer: ArrayBuffer) => {
          if (!firstAudioReceived) {
            firstAudioReceived = true;
            this.setSpeaking();
          }
          this.sendBinaryToBrowser(pcmBuffer);
        },

        onSpoken: (fullText: string) => {
          this.addToHistory("assistant", fullText);
        },

        onDone: () => {},
      });

      // After greeting, start listening
      this.isGreeting = false;
      this.activeElevenLabsClose = null;
      this.resumeListening();
    } catch (err) {
      console.error("[VoiceSession] Greeting error:", err);
      this.isGreeting = false;
      this.setState("listening");
    }
  }

  // -------------------------------------------------------------------------
  // Private: LLM → TTS Pipeline
  // -------------------------------------------------------------------------

  /**
   * Called when Deepgram returns a final transcript.
   * Triggers the streaming pipeline: LLM → TokenBuffer → ElevenLabs → browser.
   */
  private async handleFinalTranscript(userText: string): Promise<void> {
    // Guard: prevent concurrent pipeline runs (double transcripts, barge-in re-entry)
    if (this.pipelineRunning) {
      console.log("[VoiceSession] Pipeline already running, skipping transcript:", userText);
      return;
    }
    this.pipelineRunning = true;

    // Transition to processing
    this.setState("processing");

    const systemPrompt = this.config.buildPrompt(this.industry);
    let firstAudioReceived = false;
    this.firstAudioSentTime = 0;

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

        // onTTSReady: store close handle for barge-in
        onTTSReady: (closeFn: () => void) => {
          this.activeElevenLabsClose = closeFn;
        },

        // onAudio: send binary PCM to browser, set speaking state on first chunk
        onAudio: (pcmBuffer: ArrayBuffer) => {
          if (!firstAudioReceived) {
            firstAudioReceived = true;
            this.firstAudioSentTime = Date.now();
            this.setSpeaking();

            // Log server-side latency
            if (this.speechEndTime > 0) {
              const latency = this.firstAudioSentTime - this.speechEndTime;
              console.log(`[GloVoice] Server-side latency: ${latency}ms`);
            }
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
        this.pipelineRunning = false;
        this.sendToBrowser({ type: "end_call" });
        return; // Don't resume listening
      }

      // Resume listening for next user utterance
      this.activeElevenLabsClose = null;
      this.pipelineRunning = false;
      this.resumeListening();
    } catch (err) {
      console.error("[VoiceSession] Pipeline error:", err);
      this.sendToBrowser({
        type: "error",
        message: "Processing error",
      });
      this.activeElevenLabsClose = null;
      this.pipelineRunning = false;
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
