/**
 * GloVoice — Standalone voice demo service for GloMatrix landing page
 * Stack: Bun + Hono | OpenRouter (LLM) + ElevenLabs TTS
 * Browser does STT via Web Speech API (free), we handle LLM + TTS
 */

import { Hono } from "hono";
import { cors } from "hono/cors";
import { stream } from "hono/streaming";
import OpenAI from "openai";

const app = new Hono();

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "https://glomatrix.app,https://www.glomatrix.app,http://localhost:5173").split(",");
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || "";
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM"; // default: Rachel
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || "";
const LLM_MODEL = process.env.LLM_MODEL || "openai/gpt-4o-mini";

if (!ELEVENLABS_API_KEY) console.warn("[GloVoice] ELEVENLABS_API_KEY not set");
if (!OPENROUTER_API_KEY) console.warn("[GloVoice] OPENROUTER_API_KEY not set");

// OpenAI SDK works with OpenRouter — just swap the base URL
const openai = new OpenAI({
  apiKey: OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
  defaultHeaders: {
    "HTTP-Referer": "https://glomatrix.app",
    "X-Title": "GloMatrix Voice Demo",
  },
});

// System prompts per industry
const SYSTEM_PROMPTS: Record<string, string> = {
  transportation: `You are a professional AI receptionist for Glo Matrix Transportation, a logistics and trucking company. You handle driver inquiries, dispatch questions, load availability, DOT compliance questions, and client freight quotes. Be efficient and professional. Keep responses under 3 sentences.`,
  commercial: `You are a professional AI receptionist for Glo Matrix Commercial Services. You handle inquiries about janitorial, landscaping, security, and facility management services. Help callers get quotes, schedule walkthroughs, or reach the right department. Keep responses under 3 sentences.`,
  trades: `You are a professional AI receptionist for Glo Matrix HVAC & Plumbing. You handle service requests, schedule technicians, provide estimates, and answer questions about heating, cooling, and plumbing. Keep responses under 3 sentences.`,
  health: `You are a professional AI receptionist for Glo Matrix Med Spa. You help clients book appointments for facials, Botox, laser treatments, and wellness consultations. Be warm, professional, and knowledgeable. Keep responses under 3 sentences.`,
  realestate: `You are a professional AI receptionist for Glo Matrix Property Management. You handle tenant inquiries, maintenance requests, rent payment questions, and prospective resident tours. Keep responses under 3 sentences.`,
  automotive: `You are a professional AI receptionist for Glo Matrix Auto Repair. You help customers schedule oil changes, brake service, diagnostics, and other auto repair services. Provide helpful information about wait times and pricing. Keep responses under 3 sentences.`,
  beauty: `You are a professional AI receptionist for Glo Matrix Salon. You book appointments for haircuts, color, styling, and spa treatments. Be friendly and enthusiastic. Keep responses under 3 sentences.`,
  food: `You are a professional AI receptionist for Glo Matrix Catering. You handle event inquiries, menu questions, pricing, and booking for corporate and private events. Keep responses under 3 sentences.`,
  other: `You are a professional AI receptionist for Glo Matrix, an AI automation company. You help businesses understand how AI voice agents can handle their calls 24/7. Be enthusiastic and informative. Keep responses under 3 sentences.`,
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
 * Headers include X-Transcript with the text response (for accessibility)
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

  const systemPrompt = SYSTEM_PROMPTS[industry] ?? SYSTEM_PROMPTS.other;

  // Build messages array (cap history at last 6 turns to control cost)
  const recentHistory = history.slice(-6);
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
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
      max_tokens: 150,
      temperature: 0.7,
    });
    aiText = completion.choices[0]?.message?.content?.trim() ?? "I'm sorry, I didn't catch that. Could you repeat?";
  } catch (err) {
    console.error("[GloVoice] OpenAI error:", err);
    return c.json({ error: "AI service error" }, 502);
  }

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
          text: aiText,
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

  // Stream audio back to browser with AI text in header
  c.header("Content-Type", "audio/mpeg");
  c.header("X-Transcript", encodeURIComponent(aiText));
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
