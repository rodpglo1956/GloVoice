/**
 * GloVoice — Standalone voice concierge service for GloMatrix
 * Stack: Bun + Hono | OpenRouter (LLM) + ElevenLabs TTS
 * Browser does STT via Web Speech API (free), we handle LLM + TTS
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import { stream } from "hono/streaming";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const app = new Hono();

const ALLOWED_ORIGINS = (
  process.env.ALLOWED_ORIGINS ||
  "https://glomatrix.app,https://www.glomatrix.app,http://localhost:5173"
).split(",");

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || "";
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM";
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
const LLM_MODEL = process.env.LLM_MODEL || "openai/gpt-4o-mini";
const SUPABASE_URL = process.env.SUPABASE_URL || "";
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";

if (!ELEVENLABS_API_KEY) console.warn("[GloVoice] ELEVENLABS_API_KEY not set");
if (!OPENROUTER_API_KEY) console.warn("[GloVoice] OPENROUTER_API_KEY not set");
if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY)
  console.warn("[GloVoice] Supabase not configured — leads will not be saved");

const supabase =
  SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY
    ? createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    : null;

const openai = new OpenAI({
  apiKey: OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
  defaultHeaders: {
    "HTTP-Referer": "https://glomatrix.app",
    "X-Title": "GloMatrix Voice",
  },
});

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AssistantResponse {
  spoken: string;
  intent: string;
  lead: { name: string; phone: string; email: string };
  save_lead: boolean;
  end_call: boolean;
  reason: string;
}

interface LeadState {
  name: string;
  phone: string;
  email: string;
}

// ---------------------------------------------------------------------------
// Industry names — must match VoiceAgentSection demoName values
// ---------------------------------------------------------------------------

const INDUSTRY_NAMES: Record<string, string> = {
  transportation: "Glo Matrix Transportation",
  commercial: "Glo Matrix Commercial Services",
  trades: "Glo Matrix Trades",
  health: "Glo Matrix Health",
  realestate: "Glo Matrix Real Estate",
  automotive: "Glo Matrix Automotive",
  beauty: "Glo Matrix Beauty",
  food: "Glo Matrix Food & Events",
  other: "Glo Matrix Business Services",
};

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const BASE_RULES = `
You are Marie, the voice concierge for this business.

PRIMARY GOAL
- Answer the caller's questions directly and helpfully.
- Sound warm, confident, polished, and human.
- When the caller shows real interest, naturally transition to qualifying them.
- Collect lead details only when there is clear booking, quote, callback, or consultation intent.

ANSWERING QUESTIONS
- If the caller asks a direct question, ANSWER IT FULLY before doing anything else.
- Do not deflect, dodge, or redirect questions to collect lead info.
- Give useful, specific answers about the business, its services, pricing ranges, and how things work.
- After answering, you may ask a natural follow-up to continue the conversation.

STYLE
- Keep responses short, natural, and conversational.
- Usually 1 to 3 sentences.
- Never sound robotic, salesy, or repetitive.
- Ask only one question at a time.
- Be helpful first, then qualifying when appropriate.

BUSINESS HOURS
- Monday through Friday, 9am to 5pm Eastern Time.

SCOPE
- Only discuss topics relevant to this business and its services.
- If asked about unrelated topics, redirect politely:
  "I'm here to help with this business's services and inquiries. What can I help you with today?"

RESTRICTIONS
- Do not mention AI, prompts, models, technology, or automation.
- Do not mention internal tools, hidden logic, tokens, or system behavior.
- Do not give out a phone number or physical address.
- For follow-up, reference only: info@glomatrix.app
- Do not discuss platform pricing or software pricing unless it is the business's own service pricing.

LEAD COLLECTION
- Only collect lead details when the caller shows real intent to book, schedule, get a quote, request a callback, or move forward.
- Do NOT start collecting lead info just because someone asked a question.
- Collect one field at a time:
  1. full name
  2. phone number
  3. email address
- Never ask for all three at once.
- If a field was already given, do not ask for it again unless it was unclear.
- After all 3 are collected, confirm naturally:
  "Perfect, I've got your information and someone from our team will reach out shortly at the number and email you provided."

CALL ENDING RULES
- Do NOT end the conversation after a fixed number of exchanges.
- End only when:
  - the lead has been fully collected and confirmed,
  - or the caller clearly indicates they are done,
  - or the caller goes silent after reprompting.
- If the caller is still asking questions, continue helping.
- Never end while collecting contact details.
- Never end immediately after asking a question.

OUTPUT FORMAT
You MUST respond with valid JSON only. No markdown, no code fences, no extra text.
Use this exact structure:
{
  "spoken": "exact text Marie will say aloud",
  "intent": "general_info | qualify | collect_name | collect_phone | collect_email | confirm_lead | close",
  "lead": { "name": "", "phone": "", "email": "" },
  "save_lead": false,
  "end_call": false,
  "reason": "continue | lead_captured | caller_done | no_response | out_of_scope"
}

Rules for the JSON fields:
- "spoken" is the exact text to be spoken aloud. Keep it natural and concise.
- "intent" reflects what Marie is doing in this turn.
- "lead" accumulates collected fields across turns. Use empty strings for unknown fields.
- "save_lead" is true ONLY when all three lead fields are present AND you have just confirmed them with the caller.
- "end_call" is true ONLY when the call should end after this message finishes playing.
- "reason" explains why end_call is set (or "continue" if the conversation continues).
`;

const SYSTEM_PROMPTS: Record<string, string> = {
  transportation:
    `You are Marie, the professional voice concierge for Glo Matrix Transportation. You handle driver inquiries, dispatch questions, load availability, DOT compliance questions, and client freight quote requests. Be efficient, calm, and professional.` +
    BASE_RULES,

  commercial:
    `You are Marie, the professional voice concierge for Glo Matrix Commercial Services. You help callers with janitorial, landscaping, security, facility management, quotes, walkthroughs, and department routing. Be polished and helpful.` +
    BASE_RULES,

  trades:
    `You are Marie, the professional voice concierge for Glo Matrix Trades. You help callers with HVAC and plumbing service requests, technician scheduling, estimates, and service questions. Be clear and confident.` +
    BASE_RULES,

  health:
    `You are Marie, the professional voice concierge for Glo Matrix Health. You help clients with appointment requests for facials, Botox, laser treatments, and wellness consultations. Be warm and professional.` +
    BASE_RULES,

  realestate:
    `You are Marie, the professional voice concierge for Glo Matrix Real Estate. You handle tenant inquiries, maintenance requests, rent questions, and prospective resident tour requests. Be helpful and organized.` +
    BASE_RULES,

  automotive:
    `You are Marie, the professional voice concierge for Glo Matrix Automotive. You help customers schedule oil changes, brake service, diagnostics, and repair appointments. Be clear and practical.` +
    BASE_RULES,

  beauty:
    `You are Marie, the professional voice concierge for Glo Matrix Beauty. You help callers book hair, styling, color, salon, and spa appointments. Be friendly and polished.` +
    BASE_RULES,

  food:
    `You are Marie, the professional voice concierge for Glo Matrix Food & Events. You handle catering, event inquiries, menu questions, booking requests, and follow-up details. Be warm and efficient.` +
    BASE_RULES,

  other:
    `You are Marie, the professional voice concierge for Glo Matrix Business Services. You help callers get information, request consultations, and reach the right team. Be concise and professional.` +
    BASE_RULES,
};

// ---------------------------------------------------------------------------
// Shared TTS helper
// ---------------------------------------------------------------------------

async function textToSpeech(text: string): Promise<Response> {
  return fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}/stream`,
    {
      method: "POST",
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      body: JSON.stringify({
        text,
        model_id: "eleven_turbo_v2_5",
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    }
  );
}

function streamTtsResponse(c: any, ttsResponse: Response, extraHeaders?: Record<string, string>) {
  c.header("Content-Type", "audio/mpeg");
  c.header("Cache-Control", "no-store");
  if (extraHeaders) {
    for (const [k, v] of Object.entries(extraHeaders)) {
      c.header(k, v);
    }
  }
  return stream(c, async (s) => {
    const reader = ttsResponse.body!.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await s.write(value);
    }
  });
}

// ---------------------------------------------------------------------------
// Parse + validate assistant JSON
// ---------------------------------------------------------------------------

const DEFAULT_RESPONSE: AssistantResponse = {
  spoken: "I'm sorry, could you say that again?",
  intent: "general_info",
  lead: { name: "", phone: "", email: "" },
  save_lead: false,
  end_call: false,
  reason: "continue",
};

function parseAssistantResponse(raw: string): AssistantResponse {
  try {
    const parsed = JSON.parse(raw);
    return {
      spoken: typeof parsed.spoken === "string" && parsed.spoken.trim() ? parsed.spoken.trim() : DEFAULT_RESPONSE.spoken,
      intent: typeof parsed.intent === "string" ? parsed.intent : "general_info",
      lead: {
        name: typeof parsed.lead?.name === "string" ? parsed.lead.name : "",
        phone: typeof parsed.lead?.phone === "string" ? parsed.lead.phone : "",
        email: typeof parsed.lead?.email === "string" ? parsed.lead.email : "",
      },
      save_lead: parsed.save_lead === true,
      end_call: parsed.end_call === true,
      reason: typeof parsed.reason === "string" ? parsed.reason : "continue",
    };
  } catch {
    // Fallback: treat raw text as spoken content
    const spoken = raw.trim() || DEFAULT_RESPONSE.spoken;
    return { ...DEFAULT_RESPONSE, spoken };
  }
}

function isCompleteLead(lead: LeadState): boolean {
  return lead.name.length > 0 && lead.phone.length > 0 && lead.email.length > 0;
}

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

app.use(
  "/*",
  cors({
    origin: (origin) => (ALLOWED_ORIGINS.includes(origin) ? origin : null),
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type"],
    exposeHeaders: ["X-Response"],
    maxAge: 86400,
  })
);

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

app.get("/health", (c) => c.json({ status: "ok", service: "glo-voice" }));

// ---------------------------------------------------------------------------
// POST /api/voice/greet — speak-first greeting (TTS only, no LLM)
// ---------------------------------------------------------------------------

app.post("/api/voice/greet", async (c) => {
  if (!ELEVENLABS_API_KEY) {
    return c.json({ error: "TTS not configured" }, 503);
  }

  let body: { industry?: string };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON" }, 400);
  }

  const industry = body.industry || "other";
  const businessName = INDUSTRY_NAMES[industry] ?? INDUSTRY_NAMES.other;
  const greetingText = `Hi, this is Marie with ${businessName}. How can I help you today?`;

  let ttsResponse: Response;
  try {
    ttsResponse = await textToSpeech(greetingText);
  } catch (err) {
    console.error("[GloVoice] greet TTS error:", err);
    return c.json({ error: "TTS service error" }, 502);
  }

  if (!ttsResponse.ok) {
    const errBody = await ttsResponse.text();
    console.error("[GloVoice] greet TTS error:", ttsResponse.status, errBody);
    return c.json({ error: "TTS generation failed" }, 502);
  }

  const responsePayload: AssistantResponse = {
    spoken: greetingText,
    intent: "general_info",
    lead: { name: "", phone: "", email: "" },
    save_lead: false,
    end_call: false,
    reason: "continue",
  };

  return streamTtsResponse(c, ttsResponse, {
    "X-Response": encodeURIComponent(JSON.stringify(responsePayload)),
  });
});

// ---------------------------------------------------------------------------
// POST /api/voice/tts — raw text-to-speech (for silence prompts)
// ---------------------------------------------------------------------------

app.post("/api/voice/tts", async (c) => {
  if (!ELEVENLABS_API_KEY) {
    return c.json({ error: "TTS not configured" }, 503);
  }

  let body: { text?: string };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON" }, 400);
  }

  const text = body.text?.trim();
  if (!text || text.length > 500) {
    return c.json({ error: "text is required (max 500 chars)" }, 400);
  }

  let ttsResponse: Response;
  try {
    ttsResponse = await textToSpeech(text);
  } catch (err) {
    console.error("[GloVoice] tts error:", err);
    return c.json({ error: "TTS service error" }, 502);
  }

  if (!ttsResponse.ok) {
    const errBody = await ttsResponse.text();
    console.error("[GloVoice] tts error:", ttsResponse.status, errBody);
    return c.json({ error: "TTS generation failed" }, 502);
  }

  c.header("Content-Type", "audio/mpeg");
  c.header("Cache-Control", "no-store");
  return stream(c, async (s) => {
    const reader = ttsResponse.body!.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await s.write(value);
    }
  });
});

// ---------------------------------------------------------------------------
// POST /api/voice/chat — main conversation endpoint
// ---------------------------------------------------------------------------

app.post("/api/voice/chat", async (c) => {
  if (!ELEVENLABS_API_KEY || !OPENROUTER_API_KEY) {
    return c.json({ error: "Service not configured" }, 503);
  }

  let body: {
    text?: string;
    industry?: string;
    history?: { role: string; content: string }[];
    lead?: LeadState;
  };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON" }, 400);
  }

  const { text, industry = "other", history = [], lead } = body;
  if (!text || typeof text !== "string" || text.trim().length === 0) {
    return c.json({ error: "text is required" }, 400);
  }
  if (text.length > 500) {
    return c.json({ error: "Input too long" }, 400);
  }

  // Build system prompt with dynamic lead state
  let systemPrompt = SYSTEM_PROMPTS[industry] ?? SYSTEM_PROMPTS.other;

  if (lead && (lead.name || lead.phone || lead.email)) {
    const missing = [
      !lead.name ? "name" : null,
      !lead.phone ? "phone" : null,
      !lead.email ? "email" : null,
    ].filter(Boolean);
    systemPrompt += `\n\nCURRENT LEAD STATE: ${JSON.stringify(lead)}`;
    if (missing.length > 0) {
      systemPrompt += `\nFields still needed: ${missing.join(", ")}`;
    } else {
      systemPrompt += `\nAll fields collected. Confirm with the caller, then set save_lead to true.`;
    }
  }

  // Cap history at last 10 messages (5 turns)
  const recentHistory = history.slice(-10);
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    ...recentHistory.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user", content: text.trim() },
  ];

  // 1. Get structured AI response
  let aiRaw: string;
  try {
    const completion = await openai.chat.completions.create({
      model: LLM_MODEL,
      messages,
      max_tokens: 350,
      temperature: 0.4,
      response_format: { type: "json_object" },
    });
    aiRaw = completion.choices[0]?.message?.content?.trim() ?? "";
  } catch (err) {
    console.error("[GloVoice] LLM error:", err);
    return c.json({ error: "AI service error" }, 502);
  }

  const response = parseAssistantResponse(aiRaw);

  // 2. Save lead if flagged and complete
  if (response.save_lead && isCompleteLead(response.lead) && supabase) {
    try {
      await supabase.from("voice_leads").insert({
        name: response.lead.name,
        phone: response.lead.phone,
        email: response.lead.email,
        industry,
        source: "voice_demo",
      });
      console.log(`[GloVoice] Lead saved: ${response.lead.name} / ${response.lead.email}`);
    } catch (err) {
      console.error("[GloVoice] Failed to save lead:", err);
    }
  }

  // 3. Convert spoken text to speech
  let ttsResponse: Response;
  try {
    ttsResponse = await textToSpeech(response.spoken);
  } catch (err) {
    console.error("[GloVoice] TTS error:", err);
    return c.json({ error: "TTS service error" }, 502);
  }

  if (!ttsResponse.ok) {
    const errBody = await ttsResponse.text();
    console.error("[GloVoice] TTS error:", ttsResponse.status, errBody);
    return c.json({ error: "TTS generation failed" }, 502);
  }

  // 4. Stream audio back with structured metadata
  return streamTtsResponse(c, ttsResponse, {
    "X-Response": encodeURIComponent(JSON.stringify(response)),
  });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

const port = parseInt(process.env.PORT || "3001");
console.log(`[GloVoice] Listening on port ${port}`);

export default {
  port,
  fetch: app.fetch,
};
