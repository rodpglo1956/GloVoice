/**
 * Audio Pipeline — Orchestrates the streaming flow:
 *   transcript -> LLM stream -> TokenBuffer -> ElevenLabs WS -> browser audio
 *
 * Accepts all config as parameters (no shared module extraction).
 * Per-turn ElevenLabs WS (one BOS/EOS per connection).
 */

import type OpenAI from "openai";
import { TokenBuffer } from "./token-buffer";
import { SpokenExtractor } from "./spoken-extractor";
import { createElevenLabsWS } from "../ws/elevenlabs-client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LeadState {
  name: string;
  phone: string;
  email: string;
}

export interface StreamLLMToTTSParams {
  userText: string;
  systemPrompt: string;
  history: Array<{ role: "user" | "assistant"; content: string }>;
  lead: LeadState;
  openaiClient: OpenAI;
  llmModel: string;
  voiceId: string;
  elevenLabsApiKey: string;
  onAudio: (pcmBuffer: ArrayBuffer) => void;
  onSpoken: (fullText: string) => void;
  onDone: () => void;
  onTTSReady?: (closeFn: () => void) => void;
  /** Optional: called when first LLM token arrives (for stage latency logging) */
  onLLMFirstToken?: () => void;
}

export interface StreamLLMToTTSResult {
  save_lead: boolean;
  end_call: boolean;
  lead: LeadState;
}

// ---------------------------------------------------------------------------
// Parse assistant JSON
// ---------------------------------------------------------------------------

interface ParsedResponse {
  spoken: string;
  save_lead: boolean;
  end_call: boolean;
  lead: LeadState;
}

function parseStreamedResponse(raw: string): ParsedResponse {
  const fallback: ParsedResponse = {
    spoken: raw.trim() || "I'm sorry, could you say that again?",
    save_lead: false,
    end_call: false,
    lead: { name: "", phone: "", email: "" },
  };

  try {
    const cleaned = raw
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/```\s*$/, "")
      .trim();
    const parsed = JSON.parse(cleaned);
    return {
      spoken:
        typeof parsed.spoken === "string" && parsed.spoken.trim()
          ? parsed.spoken.trim()
          : fallback.spoken,
      save_lead: parsed.save_lead === true,
      end_call: parsed.end_call === true,
      lead: {
        name: parsed.lead?.name || "",
        phone: parsed.lead?.phone || "",
        email: parsed.lead?.email || "",
      },
    };
  } catch {
    return fallback;
  }
}

// ---------------------------------------------------------------------------
// Base64 -> ArrayBuffer decoder
// ---------------------------------------------------------------------------

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

// ---------------------------------------------------------------------------
// Build lead context suffix for system prompt
// ---------------------------------------------------------------------------

function buildLeadContext(lead: LeadState): string {
  if (!lead.name && !lead.phone && !lead.email) return "";

  const missing = [
    !lead.name ? "name" : null,
    !lead.phone ? "phone" : null,
    !lead.email ? "email" : null,
  ].filter(Boolean);

  let ctx = `\n\nCURRENT LEAD STATE: ${JSON.stringify(lead)}`;
  if (missing.length > 0) {
    ctx += `\nStill needed: ${missing.join(", ")}. Collect these one at a time when appropriate.`;
  } else {
    ctx += `\nAll fields collected. Confirm with the caller, set save_lead to true.`;
  }
  return ctx;
}

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------

export async function streamLLMToTTS(
  params: StreamLLMToTTSParams
): Promise<StreamLLMToTTSResult> {
  const {
    userText,
    systemPrompt,
    history,
    lead,
    openaiClient,
    llmModel,
    voiceId,
    elevenLabsApiKey,
    onAudio,
    onSpoken,
    onDone,
    onTTSReady,
    onLLMFirstToken,
  } = params;

  let fullResponse = "";

  // --- ElevenLabs WS ---
  let resolveElDone: () => void;
  const elDonePromise = new Promise<void>((resolve) => {
    resolveElDone = resolve;
  });

  const onAudioCallback = (pcmBase64: string) => {
    try {
      const buffer = base64ToArrayBuffer(pcmBase64);
      onAudio(buffer);
    } catch (err) {
      console.error("[AudioPipeline] Failed to decode audio chunk:", err);
    }
  };

  const onDoneCallback = () => {
    resolveElDone();
  };

  // New per-turn ElevenLabs WS (stream-input is one BOS/EOS per connection)
  const elevenlabs = createElevenLabsWS(
    { voiceId, apiKey: elevenLabsApiKey },
    onAudioCallback,
    onDoneCallback,
  );

  // Expose close handle for barge-in abort
  onTTSReady?.(() => elevenlabs.close());

  // --- Token Buffer ---
  const tokenBuffer = new TokenBuffer((chunk: string) => {
    elevenlabs.sendText(chunk, true);
  });

  // --- Spoken Extractor ---
  const spokenExtractor = new SpokenExtractor((text: string) => {
    tokenBuffer.add(text);
  });

  // --- LLM Streaming ---
  const fullSystemPrompt = systemPrompt + buildLeadContext(lead);
  const recentHistory = history.slice(-10);
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: fullSystemPrompt },
    ...recentHistory.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user", content: userText.trim() },
  ];

  let firstTokenFired = false;

  try {
    const stream = await openaiClient.chat.completions.create({
      model: llmModel,
      messages,
      max_tokens: 150,
      temperature: 0.6,
      top_p: 0.9,
      stream: true,
    });

    for await (const chunk of stream) {
      const token = chunk.choices[0]?.delta?.content;
      if (token) {
        if (!firstTokenFired) {
          firstTokenFired = true;
          onLLMFirstToken?.();
        }
        fullResponse += token;
        spokenExtractor.add(token);
      }
    }
  } catch (err) {
    console.error("[AudioPipeline] LLM stream error:", err);
    const fallbackText = "I'm sorry, could you say that again?";
    elevenlabs.sendText(fallbackText, true);
    fullResponse = JSON.stringify({
      spoken: fallbackText,
      save_lead: false,
      end_call: false,
    });
  }

  tokenBuffer.forceFlush();

  const parsed = parseStreamedResponse(fullResponse);

  if (!spokenExtractor.hasForwarded() && parsed.spoken) {
    console.log("[AudioPipeline] SpokenExtractor missed -- sending parsed fallback to TTS");
    elevenlabs.sendText(parsed.spoken, true);
  }

  elevenlabs.endStream();
  onSpoken(parsed.spoken);

  const timeout = new Promise<void>((resolve) => setTimeout(resolve, 15000));
  await Promise.race([elDonePromise, timeout]);

  elevenlabs.close();

  onDone();

  return {
    save_lead: parsed.save_lead,
    end_call: parsed.end_call,
    lead: parsed.lead,
  };
}
