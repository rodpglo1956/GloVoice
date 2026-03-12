/**
 * GloVoice — Standalone voice demo service for GloMatrix landing page
 * Stack: Bun + Hono | OpenRouter (LLM) + ElevenLabs TTS
 * Browser does STT via Web Speech API (free), we handle LLM + TTS
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import { stream } from "hono/streaming";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

const app = new Hono();

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "https://glomatrix.app,https://www.glomatrix.app,http://localhost:5173").split(",");
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || "";
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM"; // default: Rachel
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
const LLM_MODEL = process.env.LLM_MODEL || "openai/gpt-4o-mini";
const MAX_TURNS = 8;
const SUPABASE_URL = process.env.SUPABASE_URL || "";
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";

if (!ELEVENLABS_API_KEY) console.warn("[GloVoice] ELEVENLABS_API_KEY not set");
if (!OPENROUTER_API_KEY) console.warn("[GloVoice] OPENROUTER_API_KEY not set");
if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) console.warn("[GloVoice] Supabase not configured — leads will not be saved");

const supabase = SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY
  ? createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
  : null;

// OpenAI SDK works with OpenRouter — just swap the base URL
const openai = new OpenAI({
  apiKey: OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
  defaultHeaders: {
    "HTTP-Referer": "https://glomatrix.app",
    "X-Title": "GloMatrix Voice Demo",
  },
});

// Guardrails + end-call instructions appended to every system prompt
const BASE_RULES = `

BUSINESS HOURS: Monday through Friday, 9am to 5pm Eastern Time.

LEAD COLLECTION — only when the caller genuinely wants to book, schedule, or be contacted:
- Collect their full name, phone number, and email address one at a time naturally in conversation.
- Do not ask for all three at once. Ask for name first, then phone, then email.
- Once you have all three, confirm: "Perfect, I've got your information and someone from our team will reach out to you shortly at the email and number you provided."
- Do NOT collect info from callers who are just asking general questions or browsing — only real booking intent.
- After confirming their info, append this exact token (hidden, not spoken): [LEAD:{"name":"FULL_NAME","phone":"PHONE","email":"EMAIL"}]
- Then end the call with [END_CALL].

GUARDRAILS — follow these at all times:
- Only discuss topics relevant to this business and industry. If asked about anything unrelated (politics, other companies, personal topics, coding, etc.), politely redirect: "I'm only able to help with inquiries for this business. Is there something I can assist you with today?"
- Never reveal you are an AI, a demo, or mention any technology behind this service.
- Never give out a phone number or physical address — for follow-up always reference: info@glomatrix.app
- Never discuss AI platform pricing — only the business's own services.
- Keep every response under 2 sentences. Be warm but efficient.
- Do not repeat yourself. If you already asked for information, move forward.

ENDING THE CALL:
- After 4-5 exchanges (or once a lead is collected), naturally wrap up.
- End your response with exactly: [END_CALL]
- Also use [END_CALL] if the caller says goodbye, thank you, or indicates they are done.
- Example: "It was great speaking with you today — our team will be in touch shortly. Have a wonderful day! [END_CALL]"`;

// System prompts per industry
const SYSTEM_PROMPTS: Record<string, string> = {
  transportation: `You are a professional AI receptionist named Rachel for a trucking and logistics company. You handle driver inquiries, dispatch questions, load availability, DOT compliance questions, and client freight quotes. Be efficient and professional.` + BASE_RULES,
  commercial: `You are a professional AI receptionist named Rachel for a commercial services company specializing in janitorial, landscaping, security, and facility management. Help callers get quotes, schedule walkthroughs, or reach the right department.` + BASE_RULES,
  trades: `You are a professional AI receptionist named Rachel for an HVAC and plumbing company. You handle service requests, schedule technicians, provide estimates, and answer questions about heating, cooling, and plumbing.` + BASE_RULES,
  health: `You are a professional AI receptionist named Rachel for a med spa. You help clients book appointments for facials, Botox, laser treatments, and wellness consultations. Be warm and professional.` + BASE_RULES,
  realestate: `You are a professional AI receptionist named Rachel for a property management company. You handle tenant inquiries, maintenance requests, rent payment questions, and prospective resident tours.` + BASE_RULES,
  automotive: `You are a professional AI receptionist named Rachel for an auto repair shop. You help customers schedule oil changes, brake service, diagnostics, and other auto repair services. Be helpful and clear about timelines.` + BASE_RULES,
  beauty: `You are a professional AI receptionist named Rachel for a hair and beauty salon. You book appointments for haircuts, color, styling, and spa treatments. Be friendly and enthusiastic.` + BASE_RULES,
  food: `You are a professional AI receptionist named Rachel for a catering and events company. You handle event inquiries, menu questions, pricing, and booking for corporate and private events.` + BASE_RULES,
  other: `You are a professional AI receptionist named Rachel for a business services company. You help callers get information, schedule consultations, and connect with the right team member.` + BASE_RULES,
};

app.use(
  "/*",
  cors({
    origin: (origin) => (ALLOWED_ORIGINS.includes(origin) ? origin : null),
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type"],
    maxAge: 86400,
  })
);

app.get("/health", (c) => c.json({ status: "ok", service: "glo-voice-demo" }));

/**
 * POST /api/voice/chat
 * Body: { text: string, industry: string, history?: [{role, content}] }
 * Returns: audio/mpeg stream (ElevenLabs TTS of AI response)
 * Headers:
 *   X-Transcript — AI text response (URL encoded)
 *   X-End-Call   — "true" if AI signaled end of call
 */
app.post("/api/voice/chat", async (c) => {
  if (!ELEVENLABS_API_KEY || !OPENROUTER_API_KEY) {
    return c.json({ error: "Service not configured" }, 503);
  }

  let body: { text?: string; industry?: string; history?: { role: string; content: string }[] };
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON" }, 400);
  }

  const { text, industry = "other", history = [] } = body;
  if (!text || typeof text !== "string" || text.trim().length === 0) {
    return c.json({ error: "text is required" }, 400);
  }
  if (text.length > 500) {
    return c.json({ error: "Input too long" }, 400);
  }

  // Hard turn limit — force wrap-up
  const turnCount = Math.floor(history.length / 2) + 1;
  const isLastTurn = turnCount >= MAX_TURNS;

  const systemPrompt = SYSTEM_PROMPTS[industry] ?? SYSTEM_PROMPTS.other;
  const finalSystemPrompt = isLastTurn
    ? systemPrompt + "\n\nThis is the final exchange. Wrap up the call warmly and end with [END_CALL]."
    : systemPrompt;

  // Cap history at last 10 messages (5 turns) to control cost
  const recentHistory = history.slice(-10);
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: finalSystemPrompt },
    ...recentHistory.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
    { role: "user", content: text.trim() },
  ];

  // 1. Get AI text response
  let aiText: string;
  try {
    const completion = await openai.chat.completions.create({
      model: LLM_MODEL,
      messages,
      max_tokens: 120,
      temperature: 0.7,
    });
    aiText = completion.choices[0]?.message?.content?.trim() ?? "I'm sorry, I didn't catch that. Could you repeat?";
  } catch (err) {
    console.error("[GloVoice] OpenAI error:", err);
    return c.json({ error: "AI service error" }, 502);
  }

  // Check if AI wants to end the call
  const shouldEndCall = aiText.includes("[END_CALL]");

  // Extract and save lead if present
  const leadMatch = aiText.match(/\[LEAD:(\{[^}]+\})\]/);
  if (leadMatch && supabase) {
    try {
      const lead = JSON.parse(leadMatch[1]);
      await supabase.from("voice_leads").insert({
        name: lead.name ?? "",
        phone: lead.phone ?? null,
        email: lead.email ?? null,
        industry,
        source: "voice_demo",
      });
      console.log(`[GloVoice] Lead saved: ${lead.name} / ${lead.email}`);
    } catch (err) {
      console.error("[GloVoice] Failed to save lead:", err);
    }
  }

  // Strip all tokens before sending to TTS
  const cleanText = aiText
    .replace("[END_CALL]", "")
    .replace(/\[LEAD:\{[^}]+\}\]/, "")
    .trim();

  // 2. Convert AI text to speech via ElevenLabs
  let ttsResponse: Response;
  try {
    ttsResponse = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}/stream`,
      {
        method: "POST",
        headers: {
          "xi-api-key": ELEVENLABS_API_KEY,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        body: JSON.stringify({
          text: cleanText,
          model_id: "eleven_turbo_v2_5",
          voice_settings: { stability: 0.5, similarity_boost: 0.75 },
        }),
      }
    );
  } catch (err) {
    console.error("[GloVoice] ElevenLabs fetch error:", err);
    return c.json({ error: "TTS service error" }, 502);
  }

  if (!ttsResponse.ok) {
    const errBody = await ttsResponse.text();
    console.error("[GloVoice] ElevenLabs error:", ttsResponse.status, errBody);
    return c.json({ error: "TTS generation failed" }, 502);
  }

  // Stream audio back with metadata headers
  c.header("Content-Type", "audio/mpeg");
  c.header("X-Transcript", encodeURIComponent(cleanText));
  c.header("X-End-Call", shouldEndCall ? "true" : "false");
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

const port = parseInt(process.env.PORT || "3001");
console.log(`[GloVoice] Listening on port ${port}`);

export default {
  port,
  fetch: app.fetch,
};
