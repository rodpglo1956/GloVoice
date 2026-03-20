import { describe, test, expect } from "bun:test";
import { extractLeadFromHistory } from "./lead-extractor";

describe("extractLeadFromHistory", () => {
  // Helper to create assistant JSON entries (matches voice-session.ts addToHistory format)
  const assistant = (spoken: string) => ({
    role: "assistant" as const,
    content: JSON.stringify({ spoken, save_lead: false, end_call: false }),
  });

  const user = (text: string) => ({
    role: "user" as const,
    content: text,
  });

  // ---------------------------------------------------------------------------
  // Phone extraction
  // ---------------------------------------------------------------------------

  test("extracts phone from dashed format (770-555-1234)", () => {
    const history = [
      user("770-555-1234"),
      assistant("I've got seven seven zero. five five five. one two three four. Did I get that right?"),
    ];
    expect(extractLeadFromHistory(history).phone).toBe("7705551234");
  });

  test("extracts phone from spaced format (770 555 1234)", () => {
    const history = [user("770 555 1234")];
    expect(extractLeadFromHistory(history).phone).toBe("7705551234");
  });

  test("extracts phone from plain digits (7705551234)", () => {
    const history = [user("7705551234")];
    expect(extractLeadFromHistory(history).phone).toBe("7705551234");
  });

  test("extracts phone from parenthesized format ((770) 555-1234)", () => {
    const history = [user("(770) 555-1234")];
    expect(extractLeadFromHistory(history).phone).toBe("7705551234");
  });

  test("strips leading 1 from 11-digit phone (17705551234)", () => {
    const history = [user("1-770-555-1234")];
    expect(extractLeadFromHistory(history).phone).toBe("7705551234");
  });

  test("ignores non-phone turns (short numbers, words)", () => {
    const history = [user("yes"), user("Marcus Johnson"), user("42")];
    expect(extractLeadFromHistory(history).phone).toBe("");
  });

  // ---------------------------------------------------------------------------
  // Email extraction
  // ---------------------------------------------------------------------------

  test("extracts literal email from user turn", () => {
    const history = [user("marcus@email.com")];
    expect(extractLeadFromHistory(history).email).toBe("marcus@email.com");
  });

  test("normalizes 'at' and 'dot' in STT transcription", () => {
    const history = [user("marcus at email dot com")];
    expect(extractLeadFromHistory(history).email).toBe("marcus@email.com");
  });

  test("handles mixed format (marcus at email.com)", () => {
    const history = [user("marcus at email.com")];
    expect(extractLeadFromHistory(history).email).toBe("marcus@email.com");
  });

  test("handles email with numbers (test123@gmail.com)", () => {
    const history = [user("test123@gmail.com")];
    expect(extractLeadFromHistory(history).email).toBe("test123@gmail.com");
  });

  // ---------------------------------------------------------------------------
  // Name extraction
  // ---------------------------------------------------------------------------

  test("extracts name from turn before Marie's readback confirmation", () => {
    const history = [
      user("Marcus Johnson"),
      assistant("Quick confirm, is it Marcus Johnson?"),
      user("yes"),
    ];
    expect(extractLeadFromHistory(history).name).toBe("Marcus Johnson");
  });

  test("extracts single-word name", () => {
    const history = [
      user("Marcus"),
      assistant("Quick confirm, is it Marcus?"),
    ];
    expect(extractLeadFromHistory(history).name).toBe("Marcus");
  });

  test("skips turns with digits (not a name)", () => {
    const history = [
      user("7705551234"),
      assistant("Quick confirm, is it 7705551234?"),
    ];
    // Should not extract digits as a name
    expect(extractLeadFromHistory(history).name).toBe("");
  });

  test("skips turns with @ (email, not a name)", () => {
    const history = [
      user("marcus@email.com"),
      assistant("Quick confirm, is it marcus@email.com?"),
    ];
    expect(extractLeadFromHistory(history).name).toBe("");
  });

  // ---------------------------------------------------------------------------
  // Full scenario
  // ---------------------------------------------------------------------------

  test("extracts all 3 fields from realistic multi-turn conversation", () => {
    const history = [
      assistant("What kind of business are you running?"),
      user("I run a trucking company"),
      assistant("Tell me about your biggest pain point."),
      user("We miss a lot of calls after hours"),
      assistant("We can fix that. Want me to have someone reach out? I'd just need your name."),
      user("Marcus Johnson"),
      assistant("Quick confirm, is it Marcus Johnson?"),
      user("Yes"),
      assistant("And what's a good phone number?"),
      user("770-555-1234"),
      assistant("I've got seven seven zero. five five five. one two three four. Did I get that right?"),
      user("Yes"),
      assistant("And your email?"),
      user("marcus at email dot com"),
      assistant("So that's marcus. At email dot com. Did I get that right?"),
      user("Yes"),
      assistant("Perfect, I've got everything saved. Someone from our team will reach out soon."),
    ];

    const result = extractLeadFromHistory(history);
    expect(result.name).toBe("Marcus Johnson");
    expect(result.phone).toBe("7705551234");
    expect(result.email).toBe("marcus@email.com");
  });

  test("returns empty fields when no data found", () => {
    const history = [
      user("Tell me about your services"),
      assistant("We build AI systems that handle phone calls."),
    ];
    const result = extractLeadFromHistory(history);
    expect(result.name).toBe("");
    expect(result.phone).toBe("");
    expect(result.email).toBe("");
  });
});
