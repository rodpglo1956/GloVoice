import { describe, test, expect } from "bun:test";
import { TokenBuffer } from "./token-buffer";

describe("TokenBuffer", () => {
  test("Adding 'Hello world.' flushes 'Hello world.' on the period", () => {
    const flushed: string[] = [];
    const buf = new TokenBuffer((text) => flushed.push(text));
    buf.add("Hello world.");
    expect(flushed).toEqual(["Hello world."]);
  });

  test("Adding 'Hello' then ' world' then '.' flushes 'Hello world.'", () => {
    const flushed: string[] = [];
    const buf = new TokenBuffer((text) => flushed.push(text));
    buf.add("Hello");
    buf.add(" world");
    expect(flushed).toEqual([]);
    buf.add(".");
    expect(flushed).toEqual(["Hello world."]);
  });

  test("Adding 'Question? Yes.' flushes 'Question?' then 'Yes.'", () => {
    const flushed: string[] = [];
    const buf = new TokenBuffer((text) => flushed.push(text));
    buf.add("Question? Yes.");
    expect(flushed).toEqual(["Question?", " Yes."]);
  });

  test("forceFlush() returns remaining buffer content even without sentence boundary", () => {
    const flushed: string[] = [];
    const buf = new TokenBuffer((text) => flushed.push(text));
    buf.add("Partial content without ending");
    expect(flushed).toEqual([]);
    const remainder = buf.forceFlush();
    expect(remainder).toBe("Partial content without ending");
  });

  test("Empty buffer forceFlush returns empty string", () => {
    const flushed: string[] = [];
    const buf = new TokenBuffer((text) => flushed.push(text));
    const remainder = buf.forceFlush();
    expect(remainder).toBe("");
  });

  test("Multi-sentence 'One. Two! Three?' flushes three times", () => {
    const flushed: string[] = [];
    const buf = new TokenBuffer((text) => flushed.push(text));
    buf.add("One. Two! Three?");
    expect(flushed).toEqual(["One.", " Two!", " Three?"]);
  });
});
