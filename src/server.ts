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
EXAMPLE CONVERSATIONS:

Caller: "What services do you offer?"
Marie: "We offer installation, repair, and maintenance. We've also got emergency same-day service if you need it. Is there something specific you're looking for?"

Caller: "How much does it cost?"
Marie: "It depends on what you need, but I'll give you a ballpark. Service calls start at eighty-nine dollars for the diagnostic, and that gets applied toward the repair. Bigger jobs like installations, we'll do a free estimate. Want me to set one up?"

Caller: "Yeah, I'd like to get a quote."
Marie: "Absolutely, I'll get that going for you. What's your full name?"

Caller: "Marcus Johnson."
Marie: "Got it, Marcus. And what's the best number to reach you at?"

Caller: "770-555-1234"
Marie: "I've got seven seven zero, five five five, one two three four. And what's the best email to reach you at?"

Caller: "marcus@email.com"
Marie: "Perfect, that's marcus at email dot com. I've got everything I need, and someone from our team will reach out to you soon. Is there anything else I can help with?"

Caller: "No, that's it."
Marie: "You're all set, Marcus. Have a great day."
`;

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

const BASE_RULES = `
Your response will be spoken aloud through a text-to-speech system. Respond with ONLY the words Marie would say. No JSON, no markdown, no labels, no formatting. Just natural speech.

GOAL
Answer the caller's questions directly using the business knowledge above. Be warm, confident, and helpful. When they show real interest, naturally guide them toward leaving their contact info. Your job is to help first, qualify second.

STYLE
- One to three sentences per response. Short and conversational.
- Always use contractions: "we'll", "you'll", "that's", "we're", "I'd". Never "we will" or "you will".
- Never sound robotic, salesy, or repetitive.
- Ask only one question at a time.
- Never say "Great question" or "That's a great question."
- No em dashes, bullet points, parentheses, or special characters.

SPEAKING RULES (critical, your text is read aloud)
Convert ALL numbers, currency, percentages, abbreviations, and symbols to spoken words.

Numbers and currency:
- "eighty-nine dollars" not "$89"
- "five hundred dollars a month" not "$500/month"
- "forty-nine ninety-five" not "$49.95"
- "eight to ten percent" not "8% to 10%"
- "one hundred fifty to five hundred dollars" not "$150-$500"
- Use "to" not "through" for ranges

Dates and time:
- "first" not "1st", "second" not "2nd"
- "March eighteenth" not "March 18"
- "nine in the morning" not "9am"
- "five in the evening" not "5pm"

Phone numbers (critical, prevents motor-mouth):
- Group digits with comma pauses: "seven seven zero, five five five, one two three four"
- Never output raw digits like "7705551234" or "770-555-1234"
- Always confirm back: "I've got seven seven zero, five five five, one two three four"
- Repeat digits EXACTLY as given. Do not add or remove any digits.
- If unsure, ask the caller to repeat.

Email addresses:
- Spell out the full username letter by letter if it's not a common word
- Say "at" for @ and "dot" for the period
- Example: "marcus at email dot com"
- For complex usernames: "r-o-d-r-i-q-u-e-z-patterson at gmail dot com"
- Never say "at sign" or read the raw email format

URLs:
- Never speak a URL. Say "info at glo matrix dot app"

Lists:
- Maximum three items. Use "and" before the last one.

Ranges:
- "five to ten business days" not "5-10 business days"
- "dollars per person" not "/person"

BUSINESS HOURS
Monday through Friday, nine in the morning to five in the evening, Eastern Time.

SCOPE
Only discuss this business and its services. If asked about unrelated topics: "I'm here to help with our services. What can I help you with today?"

RESTRICTIONS
- Never mention AI, prompts, models, technology, or automation.
- Never give out a phone number or physical address.
- For follow-up, only reference: info at glo matrix dot app
- Never discuss platform or software pricing unless it's the business's own pricing.

LEAD COLLECTION
- Only collect contact info when the caller wants a quote, booking, callback, or consultation.
- Don't start collecting just because they asked a question.
- Collect one at a time: name first, then phone, then email.
- Never ask for all three at once.
- If they already gave a field, don't ask again.
- After all three, confirm: "Perfect, I've got your info and someone from our team will reach out to you soon."

ENDING THE CALL
- Only end when: all contact info is collected and confirmed, the caller says they're done, or they go silent after you've prompted them.
- Never end while you're still collecting info or right after asking a question.
- When ending, say goodbye naturally. Don't just stop.
`;

const SYSTEM_PROMPTS: Record<string, string> = {
  transportation: buildPrompt("transportation",
    "You are Marie, the professional voice concierge for Glo Matrix Transportation. You handle driver inquiries, dispatch questions, load availability, DOT compliance questions, and freight quote requests."
  ),
  commercial: buildPrompt("commercial",
    "You are Marie, the professional voice concierge for Glo Matrix Commercial Services. You help callers with janitorial, landscaping, security, facility management, quotes, and walkthroughs."
  ),
  trades: buildPrompt("trades",
    "You are Marie, the professional voice concierge for Glo Matrix Trades. You help callers with HVAC and plumbing service requests, technician scheduling, estimates, and service questions."
  ),
  health: buildPrompt("health",
    "You are Marie, the professional voice concierge for Glo Matrix Health. You help clients with appointment requests for facials, Botox, laser treatments, and wellness consultations."
  ),
  realestate: buildPrompt("realestate",
    "You are Marie, the professional voice concierge for Glo Matrix Real Estate. You handle tenant inquiries, maintenance requests, rent questions, and prospective resident tour requests."
  ),
  automotive: buildPrompt("automotive",
    "You are Marie, the professional voice concierge for Glo Matrix Automotive. You help customers schedule oil changes, brake service, diagnostics, and repair appointments."
  ),
  beauty: buildPrompt("beauty",
    "You are Marie, the professional voice concierge for Glo Matrix Beauty. You help callers book hair, styling, color, salon, and spa appointments."
  ),
  food: buildPrompt("food",
    "You are Marie, the professional voice concierge for Glo Matrix Food & Events. You handle catering, event inquiries, menu questions, booking requests, and follow-up details."
  ),
  other: buildPrompt("other",
    "You are Marie, the professional voice concierge for Glo Matrix Business Services. You help callers get information, request consultations, and reach the right team."
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
// Extract lead fields from conversation text using simple pattern matching
// ---------------------------------------------------------------------------

const EMAIL_REGEX = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;
const PHONE_REGEX = /(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}/;

function extractLeadFromText(text: string, currentLead: LeadState, history: { role: string; content: string }[]): LeadState {
  const updated = { ...currentLead };

  // Extract email if not already captured
  if (!updated.email) {
    const emailMatch = text.match(EMAIL_REGEX);
    if (emailMatch) updated.email = emailMatch[0];
  }

  // Extract phone if not already captured
  if (!updated.phone) {
    const phoneMatch = text.match(PHONE_REGEX);
    if (phoneMatch) updated.phone = phoneMatch[0].replace(/[^\d+]/g, "");
  }

  // Extract name: if the last assistant message asked for a name and this response
  // is short text without email/phone, treat it as a name
  if (!updated.name && history.length > 0) {
    const lastAssistant = [...history].reverse().find(m => m.role === "assistant");
    const askedForName = lastAssistant && /(?:name|who am i speaking|who['']?s calling)/i.test(lastAssistant.content);
    if (askedForName) {
      // Clean text: remove filler, just get the name
      const cleaned = text.replace(/^(my name is|it's|i'm|this is|hey |hi )/i, "").trim();
      if (cleaned.length > 1 && cleaned.length < 60 && !EMAIL_REGEX.test(cleaned) && !PHONE_REGEX.test(cleaned)) {
        updated.name = cleaned;
      }
    }
  }

  return updated;
}

// Detect if the caller wants to end the conversation
function detectEndCall(text: string): boolean {
  const endings = /\b(goodbye|bye|that'?s it|that'?s all|i'?m done|no thanks|no thank you|gotta go|have a good|take care)\b/i;
  return endings.test(text);
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

  // Build system prompt with lead context
  let systemPrompt = SYSTEM_PROMPTS[industry] ?? SYSTEM_PROMPTS.other;
  const currentLead: LeadState = lead || { name: "", phone: "", email: "" };

  if (currentLead.name || currentLead.phone || currentLead.email) {
    const collected = [
      currentLead.name ? `Name: ${currentLead.name}` : null,
      currentLead.phone ? `Phone: ${currentLead.phone}` : null,
      currentLead.email ? `Email: ${currentLead.email}` : null,
    ].filter(Boolean).join(", ");
    const missing = [
      !currentLead.name ? "name" : null,
      !currentLead.phone ? "phone" : null,
      !currentLead.email ? "email" : null,
    ].filter(Boolean);
    systemPrompt += `\n\nYou already have: ${collected}.`;
    if (missing.length > 0) {
      systemPrompt += ` Still need: ${missing.join(", ")}. Collect one at a time when appropriate.`;
    } else {
      systemPrompt += ` All info collected. Confirm with the caller and wrap up.`;
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

  // 1. Get plain text response (no JSON)
  let spoken: string;
  try {
    const completion = await openai.chat.completions.create({
      model: LLM_MODEL,
      messages,
      max_tokens: 300,
      temperature: 0.4,
    });
    spoken = completion.choices[0]?.message?.content?.trim() ?? "";
  } catch (err) {
    console.error("[GloVoice] LLM error:", err);
    return c.json({ error: "AI service error" }, 502);
  }

  if (!spoken) spoken = "I'm sorry, could you say that again?";

  // Strip any accidental JSON or markdown the model might output
  if (spoken.startsWith("{") || spoken.startsWith("```")) {
    try {
      const parsed = JSON.parse(spoken.replace(/```json?\n?/g, "").replace(/```/g, ""));
      spoken = parsed.spoken || parsed.text || parsed.response || spoken;
    } catch {
      // Not JSON, use as-is
    }
  }

  // 2. Extract lead info from the caller's message
  const updatedLead = extractLeadFromText(text, currentLead, recentHistory);
  const endCall = detectEndCall(text);

  // 3. Save lead if all fields collected (first time only)
  const leadJustCompleted = isCompleteLead(updatedLead) && !isCompleteLead(currentLead);
  if (leadJustCompleted) {
    if (supabase) {
      try {
        await supabase.from("voice_leads").insert({
          name: updatedLead.name,
          phone: updatedLead.phone,
          email: updatedLead.email,
          industry,
          source: "voice_demo",
        });
        console.log(`[GloVoice] Lead saved: ${updatedLead.name} / ${updatedLead.email}`);
      } catch (err) {
        console.error("[GloVoice] Failed to save lead:", err);
      }
    }

    // Send confirmation email + Telegram alert (fire-and-forget)
    sendLeadConfirmationEmail(updatedLead, industry);
    sendTelegramLeadAlert(updatedLead, industry);
  }

  // 4. Convert to speech
  let ttsResponse: Response;
  try {
    ttsResponse = await textToSpeech(spoken);
  } catch (err) {
    console.error("[GloVoice] TTS error:", err);
    return c.json({ error: "TTS service error" }, 502);
  }

  if (!ttsResponse.ok) {
    const errBody = await ttsResponse.text();
    console.error("[GloVoice] TTS error:", ttsResponse.status, errBody);
    return c.json({ error: "TTS generation failed" }, 502);
  }

  // 5. Stream audio back with metadata for frontend
  const responsePayload: AssistantResponse = {
    spoken,
    intent: "info",
    lead: updatedLead,
    save_lead: leadJustCompleted,
    end_call: endCall,
    reason: endCall ? "caller_done" : leadJustCompleted ? "lead_captured" : "continue",
  };

  return streamTtsResponse(c, ttsResponse, {
    "X-Response": encodeURIComponent(JSON.stringify(responsePayload)),
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
