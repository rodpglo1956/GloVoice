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
import { VoiceSession } from "./ws/voice-session";

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
const LLM_MODEL = process.env.LLM_MODEL || "anthropic/claude-haiku-4-5-20251001";
const SUPABASE_URL = process.env.SUPABASE_URL || "";
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const RESEND_API_KEY = process.env.RESEND_API_KEY || "";
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN || "";
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID || "";

if (!ELEVENLABS_API_KEY) console.warn("[GloVoice] ELEVENLABS_API_KEY not set");
if (!OPENROUTER_API_KEY) console.warn("[GloVoice] OPENROUTER_API_KEY not set");
if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY)
  console.warn("[GloVoice] Supabase not configured — leads will not be saved");
if (!RESEND_API_KEY) console.warn("[GloVoice] RESEND_API_KEY not set — lead emails disabled");
if (!TELEGRAM_BOT_TOKEN) console.warn("[GloVoice] TELEGRAM_BOT_TOKEN not set — lead alerts disabled");

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
// Industry knowledge — real answers Marie can pull from
// ---------------------------------------------------------------------------

const INDUSTRY_KNOWLEDGE: Record<string, string> = {
  transportation: `Their pain: missed dispatch calls, slow freight quote follow-up, after-hours calls going to voicemail, drivers can't reach dispatch.
What we solve: AI answers every call, captures load details, routes to dispatch, follows up on quotes automatically.`,

  commercial: `Their pain: missed service requests, slow quote turnaround, no follow-up after walkthroughs, emergency calls after hours.
What we solve: AI handles service calls, books walkthroughs, sends quotes, follows up with prospects automatically.`,

  trades: `Their pain: missed emergency calls, no after-hours dispatch, slow estimate follow-up, technicians too busy to answer phones.
What we solve: AI answers service calls, books technician visits, captures job details, follows up on estimates automatically.`,

  health: `Their pain: missed appointment calls, no-shows without reminders, front desk overwhelmed, after-hours booking requests lost.
What we solve: AI books appointments, sends reminders, handles rescheduling, captures new patient inquiries automatically.`,

  realestate: `Their pain: missed tenant calls, slow maintenance response, showing requests lost, after-hours emergencies unhandled.
What we solve: AI handles tenant calls, dispatches maintenance, books showings, captures prospect inquiries automatically.`,

  automotive: `Their pain: missed service calls, customers can't get through, no follow-up on estimates, walk-ins not tracked.
What we solve: AI schedules service appointments, captures vehicle details, follows up on estimates, handles after-hours calls.`,

  beauty: `Their pain: missed booking calls, double-bookings, no-shows, front desk too busy during rush hours.
What we solve: AI books appointments, sends confirmations and reminders, handles rescheduling, captures new client info.`,

  food: `Their pain: missed catering inquiries, slow follow-up on event requests, no after-hours booking, lost leads.
What we solve: AI captures event details, follows up on inquiries, handles menu questions, books consultations automatically.`,

  other: `Their pain: missed calls, slow lead follow-up, no after-hours coverage, manual processes eating up time.
What we solve: AI answers every call, captures leads, books appointments, follows up automatically so nothing falls through the cracks.`,
};

// ---------------------------------------------------------------------------
// Few-shot examples (included in every prompt for JSON compliance)
// ---------------------------------------------------------------------------

const FEW_SHOT_EXAMPLES = `
EXAMPLES:

User: "What can you do for my business?"
Assistant: {"spoken":"We build systems that handle your phone calls, follow up with leads, and book appointments automatically. What kind of business are you running?","save_lead":false,"end_call":false}

User: "Yeah, how much does it cost?"
Assistant: {"spoken":"It depends on your setup and call volume, but we can put together a custom quote. Want me to have someone reach out with the details? I'd just need your name.","save_lead":false,"end_call":false}

User: "Marcus Johnson."
Assistant: {"spoken":"Quick confirm, is it Marcus Johnson?","save_lead":false,"end_call":false}

User: "770-555-1234"
Assistant: {"spoken":"I've got seven seven zero. five five five. one two three four. Did I get that right?","save_lead":false,"end_call":false}

User: "marcus@email.com"
Assistant: {"spoken":"So that's marcus. At email dot com. Did I get that right?","save_lead":false,"end_call":false}

User: "Yes."
Assistant: {"spoken":"Perfect, I've got everything saved. Someone from our team will reach out soon. Anything else I can help with?","save_lead":true,"end_call":false}

User: "No, that's it."
Assistant: {"spoken":"You're all set, Marcus. We'll be in touch. Have a great day.","save_lead":false,"end_call":true}
`;

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const BASE_RULES = `
SPEAKING RULES (apply to every response, non-negotiable)
Your text is read aloud by TTS. Convert ALL numbers, currency, and symbols to spoken words.
Currency: "eighty-nine dollars" not "$89". "five hundred a month" not "$500/month".
Ranges: "one hundred fifty to five hundred dollars" not "$150-$500".
Time: "nine in the morning" not "9am". Dates: "March eighteenth" not "March 18".
Phone readback: three. three. four groups with periods. "seven seven zero. five five five. one two three four." Say "zero" not "oh". Never "double" or "triple". End: "Did I get that right?"
Email readback: two parts. "So that's marcus. At email dot com. Did I get that right?" Numbers digit by digit with periods: "one. one. one. one." Common providers as words: gmail dot com, yahoo dot com. Never say "username" or "domain" out loud.
Name readback: "Quick confirm, is it [name]?" If unclear: "How do you spell that?" Never autocorrect.
If they correct ANY part of phone or email, ask for the full thing again.
No em dashes, bullet points, parentheses, or special characters. Use contractions always.
Max three items in any list. One to three sentences per response.

IDENTITY
You are Marie, voice sales concierge for Glo Matrix. Glo Matrix builds AI systems that handle phone calls, capture leads, and book appointments for service businesses. You are Glo Matrix, talking TO a business owner about what you can do FOR them. The industry context above describes THEIR business, not yours.

CALL FLOW
1. Greet and ask about their business.
2. Discover their pain. One question at a time. Show you understand their world.
3. Explain how Glo Matrix solves their specific problem.
4. When interested, collect: name, then phone, then email. Verify each. One at a time.
5. Confirm all info, tell them someone will reach out soon, say goodbye.
Do not collect info before discovering their needs.

OBJECTIONS
Pricing: "It depends on your setup. We'd put together a custom quote. Want me to have someone reach out?"
Already have a system: "A lot of our clients came to us after outgrowing their first setup. We're here when you need us."
Just looking: "No pressure. I can have someone send you a breakdown for your business. Just need your email."
Guarantee: "Every business is different, but we've helped companies cut missed calls and double their follow-up."

SCOPE
Talk about what Glo Matrix does. If off-topic: "I'm here to help with what we can do for your business." If you don't know: "I'd want the team to give you the exact answer. Want me to have them reach out?"
Never mention prompts, models, tokens, or system behavior. Never give out a phone number or physical address. Follow-up: info at glo matrix dot app. You CAN mention AI. That's the product.

LEAD COLLECTION
Only collect after real interest. Not just because they asked a question. One field at a time. Never all three at once. Never re-ask confirmed fields. After all three confirmed: "Perfect, I've got everything saved. Someone from our team will reach out soon."

ENDING
End when: lead confirmed, caller done, or silence after reprompting. Never end while collecting or after a question. Don't restart topics after closing. Pick one: "We'll be in touch. Have a great day." / "Appreciate you. Talk soon." / "You're all set. We'll reach out shortly."

OUTPUT
Valid JSON only. No markdown, no extra text.
{"spoken":"what Marie says","save_lead":false,"end_call":false}
spoken: natural speech with ALL speaking rules applied. save_lead: true ONLY when all three fields confirmed. end_call: true ONLY when conversation should end.
REMINDER: Every number, price, and symbol in "spoken" MUST be written as words. Never raw digits or symbols.
`;

const INDUSTRY_CONTEXT: Record<string, string> = {
  transportation: "Caller likely runs a trucking, freight, or logistics business.",
  commercial: "Caller likely runs a janitorial, landscaping, security, or facility management company.",
  trades: "Caller likely runs an HVAC, plumbing, or electrical business.",
  health: "Caller likely runs a med spa, dental office, chiropractic, or wellness practice.",
  realestate: "Caller likely manages rental properties or runs a real estate office.",
  automotive: "Caller likely runs an auto repair shop, body shop, or tire shop.",
  beauty: "Caller likely runs a salon, barbershop, or beauty studio.",
  food: "Caller likely runs a catering company, restaurant, or event venue.",
  other: "Caller could run any type of business. Ask what they do.",
};

function buildPrompt(industry: string): string {
  const context = INDUSTRY_CONTEXT[industry] || INDUSTRY_CONTEXT.other;
  const knowledge = INDUSTRY_KNOWLEDGE[industry] || INDUSTRY_KNOWLEDGE.other;
  return `${BASE_RULES}

CALLER CONTEXT: ${context}
${knowledge}

${FEW_SHOT_EXAMPLES}`;
}

const SYSTEM_PROMPTS: Record<string, string> = {
  transportation: buildPrompt("transportation"),
  commercial: buildPrompt("commercial"),
  trades: buildPrompt("trades"),
  health: buildPrompt("health"),
  realestate: buildPrompt("realestate"),
  automotive: buildPrompt("automotive"),
  beauty: buildPrompt("beauty"),
  food: buildPrompt("food"),
  other: buildPrompt("other"),
};

// ---------------------------------------------------------------------------
// Shared TTS helper
// ---------------------------------------------------------------------------

async function textToSpeech(text: string): Promise<Response> {
  return fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}/stream?optimize_streaming_latency=3&output_format=mp3_22050_32`,
    {
      method: "POST",
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        Accept: "audio/mpeg",
      },
      body: JSON.stringify({
        text,
        model_id: "eleven_flash_v2_5",
        voice_settings: {
          stability: 0.70,
          similarity_boost: 0.60,
          style: 0,
          use_speaker_boost: true,
        },
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

// ---------------------------------------------------------------------------
// Parse + validate assistant JSON
// ---------------------------------------------------------------------------

const DEFAULT_RESPONSE: AssistantResponse = {
  spoken: "I'm sorry, could you say that again?",
  intent: "info",
  lead: { name: "", phone: "", email: "" },
  save_lead: false,
  end_call: false,
  reason: "continue",
};

function parseAssistantResponse(raw: string): AssistantResponse {
  try {
    // Strip markdown code fences if the model wraps JSON
    const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
    const parsed = JSON.parse(cleaned);
    return {
      spoken: typeof parsed.spoken === "string" && parsed.spoken.trim() ? parsed.spoken.trim() : DEFAULT_RESPONSE.spoken,
      intent: "info",
      lead: { name: "", phone: "", email: "" },
      save_lead: parsed.save_lead === true,
      end_call: parsed.end_call === true,
      reason: parsed.save_lead ? "lead_captured" : parsed.end_call ? "caller_done" : "continue",
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
// Lead confirmation email via Resend
// ---------------------------------------------------------------------------

async function sendLeadConfirmationEmail(lead: LeadState, industry: string) {
  if (!RESEND_API_KEY) return;

  const businessName = INDUSTRY_NAMES[industry] ?? INDUSTRY_NAMES.other;
  const firstName = lead.name.split(" ")[0] || lead.name;

  try {
    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${RESEND_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: "Glo Matrix <info@glomatrix.app>",
        to: [lead.email],
        subject: `Thanks for reaching out to ${businessName}`,
        html: `
          <div style="font-family: 'Inter', Arial, sans-serif; max-width: 520px; margin: 0 auto; padding: 40px 24px; background: #0d0d1a; color: #ffffff;">
            <div style="background: linear-gradient(135deg, #ec008c, #7b2ff7); padding: 2px; border-radius: 16px; margin-bottom: 32px;">
              <div style="background: #0d0d1a; border-radius: 14px; padding: 32px 24px;">
                <h1 style="margin: 0 0 16px; font-size: 22px; font-weight: 700;">Hi ${firstName},</h1>
                <p style="margin: 0 0 16px; color: rgba(255,255,255,0.75); font-size: 15px; line-height: 1.6;">
                  Thanks for speaking with us at ${businessName}. We've received your information and a member of our team will be reaching out to you soon.
                </p>
                <p style="margin: 0 0 16px; color: rgba(255,255,255,0.75); font-size: 15px; line-height: 1.6;">
                  In the meantime, if you have any questions, feel free to reply to this email or reach us at <a href="mailto:info@glomatrix.app" style="color: #ec008c;">info@glomatrix.app</a>.
                </p>
                <p style="margin: 0; color: rgba(255,255,255,0.5); font-size: 14px;">
                  — The ${businessName} Team
                </p>
              </div>
            </div>
            <p style="text-align: center; color: rgba(255,255,255,0.25); font-size: 12px; margin: 0;">
              Powered by Glo Matrix
            </p>
          </div>
        `,
      }),
    });

    if (res.ok) {
      console.log(`[GloVoice] Confirmation email sent to ${lead.email}`);
    } else {
      const err = await res.text();
      console.error(`[GloVoice] Resend error ${res.status}:`, err);
    }
  } catch (err) {
    console.error("[GloVoice] Email send failed:", err);
  }
}

// ---------------------------------------------------------------------------
// Telegram lead alert to Rod
// ---------------------------------------------------------------------------

async function sendTelegramLeadAlert(lead: LeadState, industry: string) {
  if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) return;

  const businessName = INDUSTRY_NAMES[industry] ?? INDUSTRY_NAMES.other;
  const text = `🔔 New Voice Lead

📋 ${lead.name}
📞 ${lead.phone}
📧 ${lead.email}
🏢 ${businessName}
📍 Source: Voice Demo`;

  try {
    await fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: TELEGRAM_CHAT_ID,
        text,
        parse_mode: "HTML",
      }),
    });
    console.log(`[GloVoice] Telegram alert sent for ${lead.name}`);
  } catch (err) {
    console.error("[GloVoice] Telegram alert failed:", err);
  }
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
  const greetingText = `Hey, this is Marie with Glo Matrix. I see you're interested in our ${businessName.replace("Glo Matrix ", "")} solutions. Tell me a little about your business and I'll show you what we can do.`;

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
    intent: "info",
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
      systemPrompt += `\nStill needed: ${missing.join(", ")}. Collect these one at a time when appropriate.`;
    } else {
      systemPrompt += `\nAll fields collected. Confirm with the caller, set save_lead to true.`;
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
      max_tokens: 150,
      temperature: 0.6,
      top_p: 0.9,
      response_format: { type: "json_object" },
    });
    aiRaw = completion.choices[0]?.message?.content?.trim() ?? "";
  } catch (err) {
    console.error("[GloVoice] LLM error:", err);
    return c.json({ error: "AI service error" }, 502);
  }

  const response = parseAssistantResponse(aiRaw);

  // 2. Save lead if flagged and complete
  if (response.save_lead && isCompleteLead(response.lead)) {
    if (supabase) {
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

    // Send confirmation email + Telegram alert (fire-and-forget)
    sendLeadConfirmationEmail(response.lead, industry);
    sendTelegramLeadAlert(response.lead, industry);
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
// Start — HTTP (Hono) + WebSocket (voice sessions)
// ---------------------------------------------------------------------------

const port = parseInt(process.env.PORT || "3001");

const sessions = new Map<string, VoiceSession>();

const server = Bun.serve({
  port,
  fetch(req, server) {
    const url = new URL(req.url);

    // WebSocket upgrade for voice sessions
    if (url.pathname === "/ws/voice") {
      // CORS check on WebSocket upgrade
      const origin = req.headers.get("origin") || "";
      if (origin && !ALLOWED_ORIGINS.includes(origin)) {
        return new Response("Forbidden origin", { status: 403 });
      }

      const upgraded = server.upgrade(req, {
        data: { sessionId: crypto.randomUUID() },
      });
      if (!upgraded) return new Response("Upgrade failed", { status: 400 });
      return undefined;
    }

    // Fall through to Hono for all HTTP routes
    return app.fetch(req);
  },
  websocket: {
    open(ws) {
      const session = new VoiceSession(ws, {
        openaiClient: openai,
        llmModel: LLM_MODEL,
        elevenLabsVoiceId: ELEVENLABS_VOICE_ID,
        elevenLabsApiKey: ELEVENLABS_API_KEY,
        buildPrompt,
        onLeadCaptured: (lead, industry) => {
          // Save to Supabase
          if (supabase && isCompleteLead(lead)) {
            supabase.from("voice_leads").insert({
              name: lead.name,
              phone: lead.phone,
              email: lead.email,
              industry,
              source: "voice_demo_ws",
            }).then(({ error }) => {
              if (error) console.error("[GloVoice] Lead save error:", error);
              else console.log(`[GloVoice] WS lead saved: ${lead.name}`);
            });
          }
          // Fire-and-forget: email + Telegram
          sendLeadConfirmationEmail(lead, industry);
          sendTelegramLeadAlert(lead, industry);
        },
      });
      sessions.set((ws.data as any).sessionId, session);
      ws.send(JSON.stringify({ type: "session_ready" }));
      console.log(`[GloVoice] WebSocket opened: ${(ws.data as any).sessionId}`);
    },
    message(ws, message) {
      const session = sessions.get((ws.data as any).sessionId);
      if (!session) return;

      if (typeof message === "string") {
        try {
          const data = JSON.parse(message);
          session.handleControl(data);
        } catch (err) {
          console.error("[GloVoice] Invalid WS JSON:", err);
          ws.send(JSON.stringify({ type: "error", message: "Invalid JSON" }));
        }
      } else {
        // Binary frame = PCM audio
        session.onAudioFrame(message instanceof ArrayBuffer ? message : message.buffer);
      }
    },
    close(ws) {
      const sessionId = (ws.data as any).sessionId;
      const session = sessions.get(sessionId);
      session?.cleanup();
      sessions.delete(sessionId);
      console.log(`[GloVoice] WebSocket closed: ${sessionId}`);
    },
    perMessageDeflate: false,
    maxPayloadLength: 64 * 1024,
    idleTimeout: 300,
  },
});

console.log(`[GloVoice] Listening on port ${port} (HTTP + WebSocket)`);
