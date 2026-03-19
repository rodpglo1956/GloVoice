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
  transportation: `
SERVICES OFFERED:
- Full truckload (FTL) and less-than-truckload (LTL) freight hauling
- Dedicated contract carriage for recurring routes
- Expedited and time-critical deliveries
- Flatbed, dry van, and reefer trailer options
- Real-time shipment tracking and dispatch coordination

TYPICAL PRICING:
- Freight quotes are based on mileage, weight, equipment type, and delivery timeline
- Spot rates and contract rates available
- Free quote turnaround within 2 hours during business hours

HOW IT WORKS:
1. Caller requests a quote with pickup/delivery locations and cargo details
2. Dispatch team reviews and provides a rate within 2 hours
3. Once accepted, a driver is assigned and pickup is scheduled
4. Real-time tracking provided from pickup to delivery

COMMON QUESTIONS:
- "We handle loads across the lower 48 states"
- "All drivers are DOT compliant with current medical cards and clean CSA scores"
- "We carry $1M cargo insurance on every load"
- "Yes, we can handle hazmat with properly endorsed drivers"
`,

  commercial: `
SERVICES OFFERED:
- Commercial janitorial and office cleaning (daily, weekly, or custom schedules)
- Landscaping and grounds maintenance
- Security guard staffing and patrol services
- Facility management and building maintenance
- Pressure washing, window cleaning, and exterior maintenance

TYPICAL PRICING:
- Janitorial starts at around $500/month for small offices, scales with square footage
- Landscaping maintenance starts at around $300/month
- Security staffing quoted per hour based on coverage needs
- Free on-site walkthrough and custom quote for all services

HOW IT WORKS:
1. Caller describes their facility and service needs
2. We schedule a free walkthrough to assess the property
3. A detailed proposal with pricing is delivered within 48 hours
4. Service begins within one week of signed agreement

COMMON QUESTIONS:
- "We serve commercial properties, office buildings, retail centers, and HOA communities"
- "All staff are background-checked, insured, and uniformed"
- "Yes, we offer bundled discounts when you combine multiple services"
- "Emergency cleanup and after-hours service available"
`,

  trades: `
SERVICES OFFERED:
- HVAC installation, repair, and seasonal maintenance
- Plumbing repair, repiping, water heater installation
- Electrical panel upgrades, wiring, and troubleshooting
- Preventive maintenance plans for residential and commercial
- Emergency same-day service available

TYPICAL PRICING:
- Service calls start at $89 diagnostic fee (applied toward repair)
- HVAC tune-ups around $129, full system installs vary by unit size
- Plumbing repairs typically range $150 to $500 depending on scope
- Free estimates on installations and larger projects

HOW IT WORKS:
1. Caller describes the issue or service needed
2. We schedule a technician visit (same-day for emergencies)
3. Tech diagnoses on-site and provides upfront pricing before any work
4. Work completed and warranty documentation provided

COMMON QUESTIONS:
- "All technicians are licensed, insured, and background-checked"
- "We offer financing on installations over $1,000"
- "Yes, we service both residential and commercial properties"
- "Our maintenance plans include priority scheduling and discounted rates"
`,

  health: `
SERVICES OFFERED:
- Facials, chemical peels, and microdermabrasion
- Botox and dermal filler consultations and injections
- Laser hair removal and skin resurfacing
- IV therapy and wellness drips
- Body contouring and non-invasive fat reduction

TYPICAL PRICING:
- Facials start at $95 for a basic treatment
- Botox typically $12 to $15 per unit
- Laser hair removal packages start around $200 per area
- Free consultations available for all treatments

HOW IT WORKS:
1. Caller books a free consultation or specific treatment
2. A licensed practitioner reviews goals and medical history
3. Treatment plan is customized and pricing confirmed upfront
4. Appointments scheduled at the client's convenience

COMMON QUESTIONS:
- "All treatments performed by licensed medical professionals"
- "We offer package pricing and membership plans for recurring treatments"
- "Yes, we provide before-and-after photos during consultation"
- "Most treatments have minimal downtime — you can return to normal activities the same day"
`,

  realestate: `
SERVICES OFFERED:
- Residential property management (single-family, multi-family, apartments)
- Tenant screening, lease management, and rent collection
- Maintenance coordination and emergency repair dispatch
- Property marketing, showings, and vacancy filling
- Monthly owner reporting and financial statements

TYPICAL PRICING:
- Management fees typically 8% to 10% of monthly rent collected
- Leasing fee for new tenant placement (usually one month's rent or 50%)
- Maintenance coordinated at cost with no markup
- Free property evaluation and management proposal

HOW IT WORKS:
1. Property owner contacts us for a free evaluation
2. We assess the property and present a management proposal
3. Once signed, we handle marketing, screening, leasing, and day-to-day management
4. Owners receive monthly statements and direct deposit rent payments

COMMON QUESTIONS:
- "We handle everything from finding tenants to handling midnight maintenance calls"
- "Tenant screening includes credit, background, income verification, and rental history"
- "Yes, we manage properties across the metro area"
- "Owners have 24/7 access to their property dashboard and financials"
`,

  automotive: `
SERVICES OFFERED:
- Oil changes, tire rotations, and fluid services
- Brake inspection and replacement
- Engine diagnostics and check engine light diagnosis
- Transmission service and repair
- AC repair, battery replacement, and electrical diagnostics
- State inspection and emissions testing

TYPICAL PRICING:
- Synthetic oil change starting at $49.95
- Brake pad replacement from $149 per axle
- Diagnostic fee $89 (applied toward repair if you proceed)
- Free estimates on all major repairs

HOW IT WORKS:
1. Caller describes the issue or schedules a service
2. Vehicle is inspected and a detailed estimate is provided before any work
3. No work performed without customer approval
4. Most standard services completed same-day

COMMON QUESTIONS:
- "We work on all makes and models, domestic and import"
- "ASE-certified mechanics on every job"
- "Yes, we offer loaner vehicles for repairs over 4 hours"
- "We warranty all parts and labor for 12 months or 12,000 miles"
`,

  beauty: `
SERVICES OFFERED:
- Haircuts, blowouts, and styling for all hair types
- Color services (full color, highlights, balayage, ombre)
- Keratin treatments and deep conditioning
- Braids, locs, extensions, and protective styles
- Manicures, pedicures, and nail art
- Waxing and brow shaping

TYPICAL PRICING:
- Haircuts starting at $35 (varies by length and complexity)
- Full color from $85, highlights from $120
- Braids and protective styles from $75 depending on style
- Mani/pedi combos from $55

HOW IT WORKS:
1. Caller books an appointment for their desired service
2. Consultation at the chair to confirm style, color, or treatment
3. Service performed by a licensed stylist
4. Aftercare instructions and rebooking offered

COMMON QUESTIONS:
- "We work with all hair types and textures"
- "Walk-ins welcome but appointments are recommended"
- "Yes, we do bridal and event styling — book early for those"
- "We use professional-grade products and can recommend take-home care"
`,

  food: `
SERVICES OFFERED:
- Full-service catering for corporate events, weddings, and private parties
- Custom menu planning and tastings
- Buffet, plated, and family-style service options
- Event coordination and setup/breakdown
- Weekly meal prep and delivery for individuals and offices

TYPICAL PRICING:
- Catering starts at $25 per person for buffet-style
- Plated dinner service from $45 per person
- Custom menus quoted based on guest count and selections
- Free tasting for events over 50 guests

HOW IT WORKS:
1. Caller describes the event (date, guest count, style, dietary needs)
2. We build a custom menu proposal within 48 hours
3. Tasting scheduled to finalize selections
4. Full setup, service, and breakdown handled day-of

COMMON QUESTIONS:
- "We accommodate dietary restrictions including vegan, gluten-free, kosher, and halal"
- "Yes, we provide all tableware, linens, and serving equipment"
- "Minimum 20 guests for catering events"
- "We recommend booking at least 3 weeks out for large events"
`,

  other: `
SERVICES OFFERED:
- Professional phone answering and call handling for businesses
- Appointment scheduling and reminders
- Lead capture and qualification
- Customer service and support handling
- Back-office operations and workflow management

TYPICAL PRICING:
- Service plans are customized based on call volume and features needed
- Free consultation and demo available
- No long-term contracts required

HOW IT WORKS:
1. Business owner contacts us for a free consultation
2. We assess current operations and identify opportunities to streamline
3. Custom solution is built and tested
4. Deployment with ongoing monitoring and optimization

COMMON QUESTIONS:
- "Our team handles calls around the clock so you never miss a lead"
- "Setup typically takes 5 to 10 business days"
- "Yes, the system works with your existing phone number"
- "You get a dashboard to see every call, lead, and conversation"
`,
};

// ---------------------------------------------------------------------------
// Few-shot examples (included in every prompt for JSON compliance)
// ---------------------------------------------------------------------------

const FEW_SHOT_EXAMPLES = `
EXAMPLE EXCHANGES:

User: "What can you do for my business?"
Assistant: {"spoken":"We build systems that handle your phone calls, follow up on leads, and book appointments automatically, so you're not losing business after hours or when your team's busy. What kind of business are you running?","intent":"info","lead":{"name":"","phone":"","email":""},"save_lead":false,"end_call":false,"reason":"continue"}

User: "I run a trucking company."
Assistant: {"spoken":"Nice, we work with a lot of transportation companies. Most of them come to us because they're missing dispatch calls or losing freight quotes to slow follow-up. Is that something you're dealing with?","intent":"qualify","lead":{"name":"","phone":"","email":""},"save_lead":false,"end_call":false,"reason":"continue"}

User: "Yeah, how much does it cost?"
Assistant: {"spoken":"It depends on your call volume, but our plans start at a few hundred a month. We'd put together a custom quote based on your setup. Want me to have someone reach out with the details? I'd just need your name.","intent":"qualify","lead":{"name":"","phone":"","email":""},"save_lead":false,"end_call":false,"reason":"continue"}

User: "Marcus Johnson."
Assistant: {"spoken":"Got it, Marcus. And what's the best number to reach you at?","intent":"collect_phone","lead":{"name":"Marcus Johnson","phone":"","email":""},"save_lead":false,"end_call":false,"reason":"continue"}

User: "770-555-1234"
Assistant: {"spoken":"I've got seven seven zero, five five five, one two three four. And what's the best email?","intent":"collect_email","lead":{"name":"Marcus Johnson","phone":"770-555-1234","email":""},"save_lead":false,"end_call":false,"reason":"continue"}

User: "marcus@email.com"
Assistant: {"spoken":"Perfect, that's marcus at email dot com. I've got everything I need, and someone from our team will reach out to you soon. Is there anything else I can help with?","intent":"confirm_lead","lead":{"name":"Marcus Johnson","phone":"770-555-1234","email":"marcus@email.com"},"save_lead":true,"end_call":false,"reason":"lead_captured"}

User: "No, that's it."
Assistant: {"spoken":"You're all set, Marcus. Have a great day.","intent":"close","lead":{"name":"Marcus Johnson","phone":"770-555-1234","email":"marcus@email.com"},"save_lead":false,"end_call":true,"reason":"caller_done"}
`;

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const BASE_RULES = `
You are Marie, the voice sales concierge for Glo Matrix. Glo Matrix builds AI-powered phone handling, lead capture, and appointment booking systems for service businesses. You are NOT the caller's business. You are Glo Matrix, talking TO a business owner about what Glo Matrix can do FOR their industry.

The INDUSTRY KNOWLEDGE above describes the caller's industry, not your company. Use it to show you understand THEIR world so you can sell them on how Glo Matrix solves THEIR problems.

GOAL
- Help the caller understand what Glo Matrix does for businesses like theirs.
- Show you understand their industry pain points.
- When they're interested, collect their contact info so someone from the team can follow up.

STYLE
- One to three sentences. Short, conversational, confident.
- Always use contractions: "we'll", "you'll", "that's", "we're".
- Never robotic, salesy, or repetitive.
- One question at a time.
- Never say "Great question" or "That's a great question."
- No em dashes, bullet points, or parentheses.

SPEAKING RULES (critical, your text is read aloud by TTS)
Convert ALL numbers, currency, percentages, abbreviations to spoken words.

Currency: "eighty-nine dollars" not "$89". "five hundred a month" not "$500/month". "forty-nine ninety-five" not "$49.95".
Percentages: "eight to ten percent" not "8% to 10%".
Ranges: "one hundred fifty to five hundred dollars" not "$150-$500". Use "to" not "through".
Dates: "first" not "1st". "March eighteenth" not "March 18".
Time: "nine in the morning" not "9am". "five in the evening" not "5pm".

Phone numbers (critical):
- Group with commas: "seven seven zero, five five five, one two three four"
- Never raw digits like "7705551234" or "770-555-1234"
- Confirm back: "I've got seven seven zero, five five five, one two three four"
- Repeat digits EXACTLY. Don't add or remove any. If unsure, ask them to repeat.

Emails:
- Say "at" for @, "dot" for period
- "marcus at email dot com"
- Complex usernames spell out: "r-o-d-r-i-q-u-e-z-patterson at gmail dot com"

URLs: Never speak a URL. Say "info at glo matrix dot app".
Lists: Max three items. "and" before the last one.

BUSINESS HOURS
Monday through Friday, nine in the morning to five in the evening, Eastern Time.

SCOPE
Talk about what Glo Matrix does for businesses. If asked about unrelated topics: "I'm here to help you learn about what we can do for your business. What can I tell you?"

RESTRICTIONS
- Never mention prompts, models, tokens, or system behavior.
- Never give out a phone number or physical address.
- For follow-up: info at glo matrix dot app
- You CAN mention AI when describing what Glo Matrix builds. That's the product.

LEAD COLLECTION
- Collect contact info when the caller shows interest in learning more, getting a quote, or having someone follow up.
- Don't start collecting just because they asked a question.
- One at a time: name, then phone, then email.
- Never all three at once. Don't re-ask fields already given.
- After all three: "Perfect, I've got your info and someone from our team will reach out to you soon."

CALL ENDING
- End when: lead confirmed, caller says they're done, or silence after reprompting.
- Never end while collecting info or right after a question.

OUTPUT FORMAT
Respond with valid JSON only. No markdown, no code fences.
{
  "spoken": "what Marie says aloud",
  "intent": "info | qualify | collect_name | collect_phone | collect_email | confirm_lead | close",
  "lead": { "name": "", "phone": "", "email": "" },
  "save_lead": false,
  "end_call": false,
  "reason": "continue | lead_captured | caller_done | no_response"
}
Rules:
- "spoken": natural speech, one to three sentences, all speaking rules applied
- "lead": accumulates across turns, empty string for unknown fields
- "save_lead": true ONLY when all 3 fields present AND confirmed
- "end_call": true ONLY when conversation should end
`;

const SYSTEM_PROMPTS: Record<string, string> = {
  transportation: buildPrompt("transportation",
    "The caller clicked Transportation. They likely run a trucking, freight, or logistics business. Use the industry knowledge below to show you understand their world, then explain how Glo Matrix can handle their dispatch calls, freight quote requests, and driver inquiries automatically."
  ),
  commercial: buildPrompt("commercial",
    "The caller clicked Commercial Services. They likely run a janitorial, landscaping, security, or facility management company. Use the industry knowledge below to show you understand their world, then explain how Glo Matrix can handle their service calls, quote requests, and client follow-ups automatically."
  ),
  trades: buildPrompt("trades",
    "The caller clicked Trades. They likely run an HVAC, plumbing, or electrical business. Use the industry knowledge below to show you understand their world, then explain how Glo Matrix can handle their service calls, emergency dispatch, and appointment booking automatically."
  ),
  health: buildPrompt("health",
    "The caller clicked Health & Wellness. They likely run a med spa, dental office, chiropractic clinic, or wellness practice. Use the industry knowledge below to show you understand their world, then explain how Glo Matrix can handle their appointment bookings, consultations, and follow-ups automatically."
  ),
  realestate: buildPrompt("realestate",
    "The caller clicked Real Estate. They likely manage rental properties or run a real estate office. Use the industry knowledge below to show you understand their world, then explain how Glo Matrix can handle tenant calls, maintenance requests, and showing inquiries automatically."
  ),
  automotive: buildPrompt("automotive",
    "The caller clicked Automotive. They likely run an auto repair shop, body shop, or tire shop. Use the industry knowledge below to show you understand their world, then explain how Glo Matrix can handle their service scheduling, estimate requests, and customer follow-ups automatically."
  ),
  beauty: buildPrompt("beauty",
    "The caller clicked Beauty & Care. They likely run a salon, barbershop, or beauty studio. Use the industry knowledge below to show you understand their world, then explain how Glo Matrix can handle their appointment booking, client reminders, and walk-in management automatically."
  ),
  food: buildPrompt("food",
    "The caller clicked Food & Hospitality. They likely run a catering company, restaurant, or event venue. Use the industry knowledge below to show you understand their world, then explain how Glo Matrix can handle their event inquiries, menu questions, and booking follow-ups automatically."
  ),
  other: buildPrompt("other",
    "The caller clicked Other. They could run any type of business. Ask what kind of business they have, then explain how Glo Matrix can handle their calls, lead capture, and follow-ups automatically."
  ),
};

function buildPrompt(industry: string, identity: string): string {
  return `${identity}

BUSINESS KNOWLEDGE:
${INDUSTRY_KNOWLEDGE[industry] || INDUSTRY_KNOWLEDGE.other}

${BASE_RULES}

${FEW_SHOT_EXAMPLES}`;
}

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
    const parsed = JSON.parse(raw);
    return {
      spoken: typeof parsed.spoken === "string" && parsed.spoken.trim() ? parsed.spoken.trim() : DEFAULT_RESPONSE.spoken,
      intent: typeof parsed.intent === "string" ? parsed.intent : "info",
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
      max_tokens: 400,
      temperature: 0.3,
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
// Start
// ---------------------------------------------------------------------------

const port = parseInt(process.env.PORT || "3001");
console.log(`[GloVoice] Listening on port ${port}`);

export default {
  port,
  fetch: app.fetch,
};
