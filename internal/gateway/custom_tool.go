package gateway

import (
	"encoding/json"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/yetone/magpie/internal/lazyre"
)

// Codex offers custom tools on Responses (exec in code mode, apply_patch):
// a tool of type custom whose call carries free-form input. A vendor's
// Responses API may take none of them, or only some: DeepSeek's answers
// 400 "Unsupported custom tool: 'exec'. Only 'apply_patch' is supported."
// to a request that only offers exec (#2656). A relay sends the tools as
// they came, so such a request is translated instead, on the same API:
// the translation offers each custom tool as a function taking its input
// (customSchema) and gives a call to one back as the custom_tool_call
// Codex runs. Once the model has turned them away it is translated from
// the start.

// customToolRefusal is an upstream turning a custom tool away: DeepSeek's
// "Unsupported custom tool: 'exec'", or the type named unsupported — as
// xAI names it, Rust's own wording, "unknown variant `custom`" (#3392). A
// refusal of anything else about a tool doesn't match.
var customToolRefusal = lazyre.New(`(?i)unsupported custom tool|custom tools? (?:is |are )?not (?:supported|allowed)|(?:unsupported|unknown|invalid) tool type[^\n{}]{0,20}?['"\x60]custom['"\x60]|tool type ['"\x60]custom['"\x60] (?:is )?(?:not supported|unsupported)|unknown variant[^\n{}]{0,20}?['"\x60]custom['"\x60]`)

// customToolRefused is how unfit remembers model turning custom tools
// away.
func customToolRefused(model string) string { return "custom tools\x00" + model }

// hasCustomTool reports whether a Responses body offers a custom tool:
// among its tools, in a namespace there, or in an input item that carries
// tools (Responses Lite's additional_tools, a tool search's output).
func hasCustomTool(body []byte) bool {
	if !bodyHas(body, `"custom"`) {
		return false
	}
	var m map[string]any
	if json.Unmarshal(body, &m) != nil {
		return false
	}
	if customIn(m["tools"]) {
		return true
	}
	items, _ := m["input"].([]any)
	for _, it := range items {
		if im, _ := it.(map[string]any); im != nil && customIn(im["tools"]) {
			return true
		}
	}
	return false
}

// customIn reports whether a list of tools has a custom one, a namespace's
// included.
func customIn(v any) bool {
	tools, _ := v.([]any)
	for _, t := range tools {
		tm, _ := t.(map[string]any)
		if tm == nil {
			continue
		}
		if tm["type"] == "custom" || tm["type"] == "namespace" && customIn(tm["tools"]) {
			return true
		}
	}
	return false
}

// customStream gives a custom tool's input as its call's arguments come,
// a piece at a time: the text of the input string in {"input":"…"} (or of
// a lone JSON string), unescaped, as the custom_tool_call_input deltas a
// client reads while the model writes a patch. What can't be read yet (an
// escape or a character cut off, a quote not known to end the input) waits
// for the next piece, and finish reads it once the arguments are all in.
// The finished item's input is still customInput's of the whole
// arguments; arguments of another shape stream nothing.
//
// A model offered the tool as a function writes its input as JSON it
// isn't held to, so the string is read as it meant it, not given up on
// (Codex in code mode saw its exec stop at a regex's \d, a Windows path's
// \U, a quote it hadn't escaped, while the model went on writing): an
// escape JSON hasn't is kept as written, a raw control character is kept,
// and a quote ends the input only when the arguments go on from it, with
// their next key or their end; any other quote is the input's.
type customStream struct {
	pend  []byte // of the input string, read and not given yet
	depth int
	str   bool   // in a string that isn't the input
	esc   bool   // after its backslash
	key   []byte // the depth-1 key being read, raw
	inKey bool   // the string being read is that key
	want  bool   // a key is next, at depth 1
	last  string // the last depth-1 key
	value bool   // the depth-1 value next is the input's
	in    bool   // in the input string
	lone  bool   // the input is a lone JSON string, not an object's
	over  bool   // the input is given, or there is none
}

// feed reads the next piece of the arguments and gives what it adds to
// the input.
func (c *customStream) feed(s string) string {
	if c.over {
		return ""
	}
	if c.in {
		c.pend = append(c.pend, s...)
		return c.take()
	}
	for i := 0; i < len(s); i++ {
		b := s[i]
		switch {
		case c.str:
			if c.inKey {
				c.key = append(c.key, b)
			}
			switch {
			case c.esc:
				c.esc = false
			case b == '\\':
				c.esc = true
			case b == '"':
				c.str = false
				if c.inKey {
					c.inKey = false
					c.last = ""
					json.Unmarshal(append([]byte{'"'}, c.key...), &c.last)
				}
			}
		case b == '"' && (c.depth == 0 || c.depth == 1 && c.value):
			c.in, c.lone = true, c.depth == 0
			c.pend = append(c.pend, s[i+1:]...)
			return c.take()
		case b == '"':
			c.str = true
			if c.depth == 1 && c.want {
				c.inKey, c.key, c.want = true, c.key[:0], false
			}
		case b == '{' || b == '[':
			c.depth++
			if c.depth == 1 && b == '{' {
				c.want = true
			}
		case b == '}' || b == ']':
			if c.depth--; c.depth <= 0 {
				c.over = true
				return ""
			}
		case b == ':' && c.depth == 1:
			c.value = c.last == "input"
		case b == ',' && c.depth == 1:
			c.want, c.value = true, false
		case b == ' ' || b == '\t' || b == '\n' || b == '\r':
		case c.depth == 0:
			c.over = true // not JSON this reads
			return ""
		case c.depth == 1:
			c.value = false // the input isn't a string
		}
	}
	return ""
}

// take unescapes what can be read of the input string so far, up to its
// closing quote.
func (c *customStream) take() string {
	n, end := c.readable(false)
	text := unquoteLax(c.pend[:n])
	c.pend = c.pend[n:]
	if end {
		c.over, c.pend = true, nil
	}
	return text
}

// finish gives the rest of the input once the arguments are all in: what
// waited on what came after it is read as the arguments ending there.
func (c *customStream) finish() string {
	if c.over || !c.in {
		c.over = true
		return ""
	}
	n, _ := c.readable(true)
	text := unquoteLax(c.pend[:n])
	c.over, c.pend = true, nil
	return text
}

// readable is how much of the input string waiting can be given, and
// whether the quote after it ends the input. Short of the arguments' end
// (atEnd), an escape or a character cut off waits, and so does a quote
// whose end isn't known yet.
func (c *customStream) readable(atEnd bool) (int, bool) {
	p := c.pend
	n := 0
	for n < len(p) {
		switch p[n] {
		case '"':
			switch c.closes(p[n+1:], atEnd) {
			case quoteEnds:
				return n, true
			case quoteWaits:
				return n, false
			}
			n++ // a quote the input has
		case '\\':
			k := escapeLen(p[n:], atEnd)
			if k == 0 {
				return n, false
			}
			n += k
		default:
			if !atEnd && !utf8.FullRune(p[n:]) {
				return n, false
			}
			_, size := utf8.DecodeRune(p[n:])
			n += size
		}
	}
	return n, false
}

const (
	quoteIn   = iota // a quote the input has
	quoteEnds        // the quote ends the input
	quoteWaits       // what follows doesn't say yet
)

// keyMax is how far a quote's next key is looked for: what goes on from
// a quote further than that without a key's end isn't a key.
const keyMax = 64

// closes is whether a quote ends the input, by what follows it (rest): a
// lone string's ends with the arguments, an object's when the arguments
// go on from it with their next key or their end.
func (c *customStream) closes(rest []byte, atEnd bool) int {
	i := skipSpace(rest, 0)
	if c.lone {
		switch {
		case i < len(rest):
			return quoteIn
		case atEnd:
			return quoteEnds
		}
		return quoteWaits
	}
	if i == len(rest) {
		return endsAt(atEnd)
	}
	switch rest[i] {
	case '}':
		// the object's end, when nothing but the arguments' end follows
		if i = skipSpace(rest, i+1); i < len(rest) {
			return quoteIn
		}
		return endsAt(atEnd)
	case ',':
		// the object's next key: a string, then its colon
		if i = skipSpace(rest, i+1); i == len(rest) {
			return endsAt(atEnd)
		}
		if rest[i] != '"' {
			return quoteIn
		}
		for j := i + 1; ; j++ {
			switch {
			case j >= len(rest):
				return endsAt(atEnd)
			case j-i > keyMax:
				return quoteIn
			case rest[j] == '\\':
				j++
			case rest[j] == '"':
				if j = skipSpace(rest, j+1); j == len(rest) {
					return endsAt(atEnd)
				}
				if rest[j] == ':' {
					return quoteEnds
				}
				return quoteIn
			}
		}
	}
	return quoteIn
}

// endsAt is a quote's end where the arguments run out after it: the end
// once they are all in, unknown till then.
func endsAt(atEnd bool) int {
	if atEnd {
		return quoteEnds
	}
	return quoteWaits
}

// skipSpace is the index of the first byte from i in s that isn't JSON's
// white space.
func skipSpace(s []byte, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
		i++
	}
	return i
}

// escapeLen is the length of the escape at the start of s, a backslash:
// JSON's own, a surrogate pair's two halves together; 1 for one JSON
// hasn't, its backslash kept as written; 0 when it is cut off, short of
// the arguments' end.
func escapeLen(s []byte, atEnd bool) int {
	if len(s) < 2 {
		if atEnd {
			return 1
		}
		return 0
	}
	switch s[1] {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		return 2
	case 'u':
	default:
		return 1
	}
	switch h := hexDigits(s[2:]); {
	case h < 4 && 2+h == len(s) && !atEnd:
		return 0 // its digits cut off
	case h < 4:
		return 1 // not an escape: \u without four digits, as written
	case !highSurrogate(s[2:6]):
		return 6
	}
	// a pair's first half: the second goes with it
	switch next := s[6:]; {
	case len(next) >= 6 && next[0] == '\\' && next[1] == 'u' && hexDigits(next[2:]) == 4:
		if lowSurrogate(next[2:6]) {
			return 12
		}
		return 6
	case !atEnd && len(next) < 6 && (len(next) == 0 || next[0] == '\\' && (len(next) == 1 || next[1] == 'u' && hexDigits(next[2:]) == len(next)-2)):
		return 0 // the second half may be cut off
	}
	return 6 // a lone half, read as JSON reads it
}

// hexDigits is how many of the first four bytes of s are hex digits, up
// to the first that isn't.
func hexDigits(s []byte) int {
	n := 0
	for n < 4 && n < len(s) && (s[n] >= '0' && s[n] <= '9' || s[n] >= 'a' && s[n] <= 'f' || s[n] >= 'A' && s[n] <= 'F') {
		n++
	}
	return n
}

// highSurrogate is whether the four hex digits of a \u escape are the
// first half of a pair.
func highSurrogate(hex []byte) bool {
	r, err := strconv.ParseUint(string(hex), 16, 16)
	return err == nil && r >= 0xD800 && r < 0xDC00
}

// lowSurrogate is whether they are the second half.
func lowSurrogate(hex []byte) bool {
	r, err := strconv.ParseUint(string(hex), 16, 16)
	return err == nil && r >= 0xDC00 && r < 0xE000
}

// unquoteLax is the text of a JSON string's contents (no quotes), read as
// JSON reads it where it can, and otherwise as the model meant it: an
// escape JSON hasn't, a raw control character, a quote, kept as written;
// bytes that aren't UTF-8 read as JSON reads them, U+FFFD.
func unquoteLax(s []byte) string {
	var text string
	if json.Unmarshal(append(append([]byte{'"'}, s...), '"'), &text) == nil {
		return text
	}
	b := make([]byte, 0, len(s))
	for i := 0; i < len(s); {
		if s[i] != '\\' {
			r, size := utf8.DecodeRune(s[i:])
			if r == utf8.RuneError && size == 1 {
				b = utf8.AppendRune(b, utf8.RuneError)
			} else {
				b = append(b, s[i:i+size]...)
			}
			i += size
			continue
		}
		k := escapeLen(s[i:], true)
		switch {
		case k == 1:
			b = append(b, '\\')
		case s[i+1] == 'u':
			r := rune(hexValue(s[i+2 : i+6]))
			if k == 12 {
				r = utf16.DecodeRune(r, rune(hexValue(s[i+8:i+12])))
			} else if utf16.IsSurrogate(r) {
				r = utf8.RuneError
			}
			b = utf8.AppendRune(b, r)
		default:
			b = append(b, jsonUnescape[s[i+1]])
		}
		i += k
	}
	return string(b)
}

// jsonUnescape is the byte each of JSON's one-letter escapes stands for.
var jsonUnescape = [256]byte{'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}

// hexValue is the value of four hex digits.
func hexValue(hex []byte) uint64 {
	v, _ := strconv.ParseUint(string(hex), 16, 16)
	return v
}
