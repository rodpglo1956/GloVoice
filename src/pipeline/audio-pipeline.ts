/**
 * Audio Pipeline — Orchestrates the streaming flow:
 *   transcript → LLM stream → TokenBuffer → ElevenLabs WS → browser audio
 *
 * Accepts all config as parameters (no shared module extraction).
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
  /** Called with decoded PCM audio buffer for each ElevenLabs chunk */
  onAudio: (pcmBuffer: ArrayBuffer) => void;
  /** Called with the full spoken text after LLM stream completes */
  onSpoken: (fullText: string) => void;
  /** Called when all audio generation is complete */
  onDone: () => void;
  /** Called with the ElevenLabs close function so caller can abort TTS on barge-in */
  onTTSReady?: (closeFn: () => void) => void;
}

export interface StreamLLMToTTSResult {
  save_lead: boolean;
  end_call: boolean;
  lead: LeadState;
}

// ---------------------------------------------------------------------------
// Parse assistant JSON (same logic as server.ts parseAssistantResponse)
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
// Base64 → ArrayBuffer decoder
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

/**
 * Stream LLM response through TokenBuffer into ElevenLabs WebSocket TTS.
 * Audio chunks are delivered to the browser as they arrive from ElevenLabs.
 *
 * Returns flags (save_lead, end_call, lead) for the caller to act on.
 */
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
  } = params;

  // Track the full accumulated LLM response for parsing after stream ends
  let fullResponse = "";

  // --- ElevenLabs WS ---
  // We need a Promise that resolves when ElevenLabs signals it's done
  let resolveElDone: () => void;
  const elDonePromise = new Promise<void>((resolve) => {
    resolveElDone = resolve;
  });

  const elevenlabs = createElevenLabsWS(
    { voiceId, apiKey: elevenLabsApiKey },
    // onAudio: decode base64 PCM → ArrayBuffer → send to browser
    (pcmBase64: string) => {
      try {
        const buffer = base64ToArrayBuffer(pcmBase64);
        onAudio(buffer);
      } catch (err) {
        console.error("[AudioPipeline] Failed to decode audio chunk:", err);
      }
    },
    // onDone: ElevenLabs finished generating all audio
    () => {
      resolveElDone();
    }
  );

  // Expose close handle for barge-in abort
  onTTSReady?.(() => elevenlabs.close());

  // --- Token Buffer ---
  const tokenBuffer = new TokenBuffer((chunk: string) => {
    // Each flushed sentence gets sent to ElevenLabs with flush=true
    elevenlabs.sendText(chunk, true);
  });

  // --- Spoken Extractor ---
  // LLM outputs JSON: {"spoken":"...","save_lead":false,...}
  // We only want to send the "spoken" value to TTS, not the JSON syntax.
  const spokenExtractor = new SpokenExtractor((text: string) => {
    tokenBuffer.add(text);
  });

  // --- LLM Streaming ---
  const fullSystemPrompt = systemPrompt + buildLeadContext(lead);

  // Cap history at last 10 messages (5 turns)
  const recentHistory = history.slice(-10);
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: fullSystemPrompt },
    ...recentHistory.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user", content: userText.trim() },
  ];

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
        fullResponse += token;
        spokenExtractor.add(token);
      }
    }
  } catch (err) {
    console.error("[AudioPipeline] LLM stream error:", err);
    // On LLM error, flush a fallback message through TTS
    const fallbackText = "I'm sorry, could you say that again?";
    elevenlabs.sendText(fallbackText, true);
    fullResponse = JSON.stringify({
      spoken: fallbackText,
      save_lead: false,
      end_call: false,
    });
  }

  // Flush any remaining tokens after LLM stream completes
  tokenBuffer.forceFlush();

  // Parse the full accumulated response for flags
  const parsed = parseStreamedResponse(fullResponse);

  // Fallback: if SpokenExtractor never found "spoken" in the stream
  // (LLM returned plain text, or JSON was malformed), send parsed spoken text directly
  if (!spokenExtractor.hasForwarded() && parsed.spoken) {
    console.log("[AudioPipeline] SpokenExtractor missed — sending parsed fallback to TTS");
    elevenlabs.sendText(parsed.spoken, true);
  }

  // Signal end of text input to ElevenLabs
  elevenlabs.endStream();

  // Notify caller of the full spoken text
  onSpoken(parsed.spoken);

  // Wait for ElevenLabs to finish generating audio, with a timeout
  const timeout = new Promise<void>((resolve) => setTimeout(resolve, 15000));
  await Promise.race([elDonePromise, timeout]);

  // Clean up ElevenLabs connection
  elevenlabs.close();

  // Notify caller that all audio is done
  onDone();

  return {
    save_lead: parsed.save_lead,
    end_call: parsed.end_call,
    lead: parsed.lead,
  };
}
