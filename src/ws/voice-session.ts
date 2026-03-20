/**
 * VoiceSession -- Per-connection state machine for streaming voice pipeline.
 * Manages Deepgram STT, per-turn ElevenLabs TTS, audio forwarding,
 * LLM-to-TTS pipeline, and transcript delivery.
 */

import { createDeepgramConnection, type DeepgramConnection } from "./deepgram-client";
import {
  streamLLMToTTS,
  type LeadState,
  type StreamLLMToTTSResult,
} from "../pipeline/audio-pipeline";
import { createVADProcessor, type VADProcessor } from "./vad-processor";
import { extractLeadFromHistory } from "../pipeline/lead-extractor";
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
  private static readonly BARGE_IN_THRESHOLD = 1; // VAD already requires 6 consecutive frames internally
  /** Active ElevenLabs context close handle for barge-in */
  private activeElevenLabsClose: (() => void) | null = null;
  /** Timestamp of last speech_final / utterance_end from Deepgram */
  private speechEndTime = 0;
  /** Timestamp when first audio chunk sent to browser for current turn */
  private firstAudioSentTime = 0;
  /** Whether this is the greeting turn (no user input yet) */
  private isGreeting = false;
  /** Guard: true while handleFinalTranscript is running */
  private pipelineRunning = false;

  constructor(
    browserWs: { send(data: string | ArrayBuffer): number },
    config: VoiceSessionConfig
  ) {
    this.browserWs = browserWs;
    this.config = config;
    this.vadProcessor = createVADProcessor();
    console.log("[VoiceSession] VAD processor initialized (energy-based)");
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

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

  onAudioFrame(data: ArrayBuffer): void {
    if (this.state === "listening" && this.deepgram) {
      this.deepgram.send(data);
    } else if (this.state === "speaking") {
      // Do NOT send audio to Deepgram during speaking — speaker echo gets
      // transcribed as user speech, causing hallucination and false responses.
      // Only run VAD for barge-in detection.
      if (this.vadProcessor) {
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
    // During 'processing' state: don't feed Deepgram either (prevents echo transcription)
  }

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

  private handleBargeIn(): void {
    if (this.state !== "speaking") return;

    console.log("[VoiceSession] Barge-in detected -- interrupting agent");

    this.setState("interrupted");

    // Close the active ElevenLabs WS to stop TTS generation
    if (this.activeElevenLabsClose) {
      this.activeElevenLabsClose();
      this.activeElevenLabsClose = null;
    }

    this.sendToBrowser({ type: "stop_playback" });

    this.vadProcessor?.reset();
    this.consecutiveSpeechFrames = 0;

    this.setState("listening");
  }

  getState(): SessionState { return this.state; }
  getHistory(): ConversationEntry[] { return this.conversationHistory; }
  getIndustry(): string { return this.industry; }

  setSpeaking(): void {
    this.setState("speaking");
    this.vadProcessor?.startCooldown(800);
    this.consecutiveSpeechFrames = 0;
  }

  resumeListening(): void {
    if (this.state === "speaking" || this.state === "processing" || this.state === "interrupted") {
      this.setState("listening");
    }
  }

  addToHistory(role: "user" | "assistant", content: string): void {
    if (role === "assistant") {
      const jsonContent = JSON.stringify({ spoken: content, save_lead: false, end_call: false });
      this.conversationHistory.push({ role, content: jsonContent });
    } else {
      this.conversationHistory.push({ role, content });
    }
    if (this.conversationHistory.length > 20) {
      this.conversationHistory = this.conversationHistory.slice(-20);
    }
  }

  // -------------------------------------------------------------------------
  // Private: Listening / STT
  // -------------------------------------------------------------------------

  private startListening(industry: string): void {
    this.industry = industry;

    if (this.deepgram) {
      this.deepgram.close();
      this.deepgram = null;
    }

    let lastHandledTranscript = "";

    this.deepgram = createDeepgramConnection(
      (text: string, isFinal: boolean) => {
        if (isFinal) {
          this.speechEndTime = Date.now();
        }

        this.sendToBrowser({
          type: "transcript",
          text,
          final: isFinal,
          ...(isFinal ? { speechEndTime: this.speechEndTime } : {}),
        });

        if (isFinal && text.trim()) {
          if (text.trim() === lastHandledTranscript) {
            console.log("[VoiceSession] Skipping duplicate final transcript");
            return;
          }
          lastHandledTranscript = text.trim();

          this.addToHistory("user", text.trim());
          this.handleFinalTranscript(text.trim());
        }
      },
      (error: Error) => {
        console.error("[VoiceSession] Deepgram error:", error.message);
        this.sendToBrowser({ type: "error", message: "Speech recognition error" });
      },
      () => {
        console.log("[VoiceSession] Deepgram connection closed");
      }
    );

    this.startKeepAlive();
    console.log(`[VoiceSession] Started listening for industry: ${industry}`);

    this.playGreeting();
  }

  // -------------------------------------------------------------------------
  // Private: Greeting
  // -------------------------------------------------------------------------

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
  // Private: LLM -> TTS Pipeline
  // -------------------------------------------------------------------------

  private async handleFinalTranscript(userText: string): Promise<void> {
    if (this.pipelineRunning) {
      console.log("[VoiceSession] Pipeline already running, skipping transcript:", userText);
      return;
    }
    this.pipelineRunning = true;

    this.setState("processing");

    const systemPrompt = this.config.buildPrompt(this.industry);
    let firstAudioReceived = false;
    this.firstAudioSentTime = 0;
    let llmFirstTokenTime = 0;

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

        onTTSReady: (closeFn: () => void) => {
          this.activeElevenLabsClose = closeFn;
        },

        onLLMFirstToken: () => {
          llmFirstTokenTime = Date.now();
          if (this.speechEndTime > 0) {
            console.log(`[GloVoice] Deepgram->LLM first token: ${llmFirstTokenTime - this.speechEndTime}ms`);
          }
        },

        onAudio: (pcmBuffer: ArrayBuffer) => {
          if (!firstAudioReceived) {
            firstAudioReceived = true;
            this.firstAudioSentTime = Date.now();
            this.setSpeaking();

            if (this.speechEndTime > 0) {
              const total = this.firstAudioSentTime - this.speechEndTime;
              const llmToTTS = llmFirstTokenTime > 0 ? this.firstAudioSentTime - llmFirstTokenTime : 0;
              console.log(`[GloVoice] LLM->TTS first audio: ${llmToTTS}ms`);
              console.log(`[GloVoice] Total server-side latency: ${total}ms`);
            }
          }
          this.sendBinaryToBrowser(pcmBuffer);
        },

        onSpoken: (fullText: string) => {
          this.addToHistory("assistant", fullText);
        },

        onDone: () => {},
      });

      // Handle lead capture -- extract from conversation history
      if (result.save_lead) {
        const extracted = extractLeadFromHistory(this.conversationHistory);
        if (extracted.name) this.lead.name = extracted.name;
        if (extracted.phone) this.lead.phone = extracted.phone;
        if (extracted.email) this.lead.email = extracted.email;

        console.log(`[VoiceSession] Lead extracted: name="${this.lead.name}" phone="${this.lead.phone}" email="${this.lead.email}"`);

        this.config.onLeadCaptured?.(this.lead, this.industry);
      }

      if (result.end_call) {
        this.pipelineRunning = false;
        this.sendToBrowser({ type: "end_call" });
        return;
      }

      this.activeElevenLabsClose = null;
      this.pipelineRunning = false;
      this.resumeListening();
    } catch (err) {
      console.error("[VoiceSession] Pipeline error:", err);
      this.sendToBrowser({ type: "error", message: "Processing error" });
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
      this.sendToBrowser({ type: "status", state: newState });
    }
  }

  private sendToBrowser(data: Record<string, unknown>): void {
    try { this.browserWs.send(JSON.stringify(data)); }
    catch (err) { console.error("[VoiceSession] Failed to send to browser:", err); }
  }

  private sendBinaryToBrowser(buffer: ArrayBuffer): void {
    try { this.browserWs.send(buffer); }
    catch (err) { console.error("[VoiceSession] Failed to send binary to browser:", err); }
  }

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
