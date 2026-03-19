/**
 * SpokenExtractor — Extracts the "spoken" field value from a streaming JSON
 * response, forwarding only the speech content to TTS (not JSON syntax).
 *
 * The LLM outputs: {"spoken":"Hey there, tell me about your business.","save_lead":false,...}
 * We only want to send "Hey there, tell me about your business." to ElevenLabs.
 *
 * State machine approach: accumulate tokens, detect when we enter the "spoken"
 * value string, forward only those characters, stop when the string closes.
 */

type State = "seeking" | "in_spoken" | "done";

export class SpokenExtractor {
  private accumulated = "";
  private state: State = "seeking";
  private onToken: (text: string) => void;
  /** Track escape sequences inside the spoken string */
  private escaped = false;
  /** Whether any content was actually forwarded to TTS */
  private forwarded = false;

  constructor(onToken: (text: string) => void) {
    this.onToken = onToken;
  }

  /**
   * Feed a token from the LLM stream.
   * Only forwards characters that are inside the "spoken" value string.
   */
  add(token: string): void {
    this.accumulated += token;

    if (this.state === "done") return;

    if (this.state === "seeking") {
      // Look for "spoken":" pattern (with possible whitespace)
      const marker = this.findSpokenStart(this.accumulated);
      if (marker >= 0) {
        this.state = "in_spoken";
        // Everything after the opening quote of the value is speech content
        const afterMarker = this.accumulated.slice(marker);
        if (afterMarker) {
          this.forwardSpoken(afterMarker);
        }
      }
      return;
    }

    if (this.state === "in_spoken") {
      // Forward the new token, checking for end of string
      this.forwardSpoken(token);
    }
  }

  /**
   * Whether any spoken content was forwarded to the callback.
   */
  hasForwarded(): boolean {
    return this.forwarded;
  }

  /**
   * Get the full accumulated LLM output (for parsing flags after stream ends).
   */
  getAccumulated(): string {
    return this.accumulated;
  }

  /**
   * Find the position right after "spoken":"  (opening quote of the value).
   * Returns the index of the first character of the spoken content, or -1.
   */
  private findSpokenStart(text: string): number {
    // Match "spoken" followed by optional whitespace, colon, optional whitespace, opening quote
    const regex = /"spoken"\s*:\s*"/;
    const match = regex.exec(text);
    if (match) {
      return match.index + match[0].length;
    }
    return -1;
  }

  /**
   * Forward characters from the spoken value string to onToken.
   * Stops at the closing unescaped quote and transitions to "done".
   */
  private forwardSpoken(text: string): void {
    let output = "";

    for (let i = 0; i < text.length; i++) {
      const ch = text[i];

      if (this.escaped) {
        // Handle escape sequences — convert JSON escapes to actual chars
        this.escaped = false;
        if (ch === "n") output += "\n";
        else if (ch === "t") output += "\t";
        else if (ch === '"') output += '"';
        else if (ch === "\\") output += "\\";
        else output += ch; // pass through unknown escapes
        continue;
      }

      if (ch === "\\") {
        this.escaped = true;
        continue;
      }

      if (ch === '"') {
        // End of spoken string
        this.state = "done";
        break;
      }

      output += ch;
    }

    if (output) {
      this.forwarded = true;
      this.onToken(output);
    }
  }
}
