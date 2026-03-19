/**
 * TokenBuffer — Accumulates LLM tokens and flushes on sentence boundaries.
 * Designed for LLM-to-TTS streaming: tokens arrive one at a time,
 * buffer flushes complete sentences to ElevenLabs for natural speech.
 *
 * Sentence boundaries: . ! ? : followed by whitespace or end-of-string.
 */

export class TokenBuffer {
  private buffer: string = "";
  private onFlush: (text: string) => void;

  constructor(onFlush: (text: string) => void) {
    this.onFlush = onFlush;
  }

  /**
   * Add a token (or chunk of text) to the buffer.
   * After appending, scan for sentence boundaries and flush completed sentences.
   */
  add(token: string): void {
    this.buffer += token;
    this.drainSentences();
  }

  /**
   * Force-flush whatever remains in the buffer (for the last partial sentence
   * when the LLM stream finishes). Returns the flushed text.
   */
  forceFlush(): string {
    const text = this.buffer;
    this.buffer = "";
    if (text) {
      this.onFlush(text);
    }
    return text;
  }

  /**
   * Reset the buffer without flushing.
   */
  clear(): void {
    this.buffer = "";
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  /**
   * Scan the buffer for sentence-ending punctuation (. ! ? :) followed by
   * whitespace or at the very end of the buffer. Flush each complete sentence.
   */
  private drainSentences(): void {
    // Match: sentence-ending punctuation followed by a space (meaning next sentence starts).
    // We split on the boundary AFTER the punctuation and BEFORE the space.
    // Also handle punctuation at end-of-string (the whole buffer is one complete sentence).
    const boundaryRegex = /[.!?](?=\s|$)/g;

    let lastFlushEnd = 0;
    let match: RegExpExecArray | null;

    while ((match = boundaryRegex.exec(this.buffer)) !== null) {
      const sentenceEnd = match.index + match[0].length;
      const chunk = this.buffer.slice(lastFlushEnd, sentenceEnd);
      if (chunk) {
        this.onFlush(chunk);
      }
      lastFlushEnd = sentenceEnd;
    }

    if (lastFlushEnd > 0) {
      this.buffer = this.buffer.slice(lastFlushEnd);
    }
  }
}
