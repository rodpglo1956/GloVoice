/**
 * Lead Extractor — Extracts name, phone, and email from conversation history.
 *
 * The LLM outputs {spoken, save_lead, end_call} only (no lead field).
 * When save_lead:true fires, this module parses the conversation transcript
 * to find the confirmed name, phone, and email.
 *
 * Assistant entries are stored as JSON: {"spoken":"...","save_lead":false,"end_call":false}
 * User entries are stored as plain text.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ConversationEntry {
  role: "user" | "assistant";
  content: string;
}

export interface ExtractedLead {
  name: string;
  phone: string;
  email: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Extract the "spoken" text from an assistant JSON entry */
function getSpoken(entry: ConversationEntry): string {
  if (entry.role !== "assistant") return "";
  try {
    const parsed = JSON.parse(entry.content);
    return typeof parsed.spoken === "string" ? parsed.spoken : "";
  } catch {
    return entry.content;
  }
}

// ---------------------------------------------------------------------------
// Extractors
// ---------------------------------------------------------------------------

/**
 * Extract phone number from user turns.
 * Handles: 770-555-1234, (770) 555-1234, 770 555 1234, 7705551234, 1-770-555-1234
 */
function extractPhone(history: ConversationEntry[]): string {
  for (const entry of history) {
    if (entry.role !== "user") continue;

    // Strip all non-digits
    const digits = entry.content.replace(/\D/g, "");

    // Must be exactly 10 or 11 digits (with leading 1)
    if (digits.length === 10) return digits;
    if (digits.length === 11 && digits.startsWith("1")) return digits.slice(1);
  }
  return "";
}

/**
 * Extract email from user turns.
 * Tries literal email first, then normalizes STT "at"/"dot" patterns.
 */
function extractEmail(history: ConversationEntry[]): string {
  const emailRegex = /[\w.+\-]+@[\w.\-]+\.\w{2,}/i;

  for (const entry of history) {
    if (entry.role !== "user") continue;

    // Try literal email
    const directMatch = emailRegex.exec(entry.content);
    if (directMatch) return directMatch[0].toLowerCase();

    // Normalize STT patterns: "at" -> "@", "dot" -> "."
    const normalized = entry.content
      .replace(/\s+at\s+/gi, "@")
      .replace(/\s+dot\s+/gi, ".")
      .replace(/\s+/g, "");

    const normalizedMatch = emailRegex.exec(normalized);
    if (normalizedMatch) return normalizedMatch[0].toLowerCase();
  }
  return "";
}

/**
 * Extract name from user turn preceding Marie's readback confirmation.
 * Pattern: user says name -> assistant says "Quick confirm, is it [name]?"
 */
function extractName(history: ConversationEntry[]): string {
  const readbackPattern = /quick confirm|is it .+\?/i;

  for (let i = 1; i < history.length; i++) {
    const entry = history[i];
    if (entry.role !== "assistant") continue;

    const spoken = getSpoken(entry);
    if (!readbackPattern.test(spoken)) continue;

    // Look at the preceding user turn
    const prevUser = history[i - 1];
    if (!prevUser || prevUser.role !== "user") continue;

    const candidate = prevUser.content.trim();

    // Skip if it contains digits (phone number) or @ (email)
    if (/\d/.test(candidate)) continue;
    if (/@/.test(candidate)) continue;

    // Must be 1-5 words
    const words = candidate.split(/\s+/);
    if (words.length < 1 || words.length > 5) continue;

    return candidate;
  }
  return "";
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Extract lead data from conversation history.
 * Returns {name, phone, email} with empty strings for unfound fields.
 */
export function extractLeadFromHistory(history: ConversationEntry[]): ExtractedLead {
  return {
    name: extractName(history),
    phone: extractPhone(history),
    email: extractEmail(history),
  };
}
