package gateway

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// gatedCall is a Responses upstream that streams a call's first argument
// delta and then waits, before the rest of the call, until the test has
// seen that delta reach the client (or gives up).
type gatedCall struct {
	name  string        // the function the upstream calls
	first string        // its first argument delta
	rest  string        // the rest of its arguments
	seen  chan struct{} // closed by the test once the client has the first delta
	wrote chan struct{} // closed by the upstream as it goes on to the rest
}

func (g *gatedCall) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	io.ReadAll(r.Body)
	w.Header().Set("Content-Type", "text/event-stream")
	f := w.(http.Flusher)
	q := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	write := func(lines ...string) {
		io.WriteString(w, sse(lines...))
		f.Flush()
	}
	write(`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"`+g.name+`","arguments":""}}`,
		`data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":`+q(g.first)+`}`)
	select {
	case <-g.seen:
	case <-time.After(20 * time.Second):
	case <-r.Context().Done():
		return
	}
	close(g.wrote)
	args := q(g.first + g.rest)
	write(`data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":`+q(g.rest)+`}`,
		`data: {"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_1","arguments":`+args+`}`,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"`+g.name+`","arguments":`+args+`,"status":"completed"}}`,
		`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[],"usage":{"input_tokens":5,"output_tokens":5}}}`)
}

// A call's arguments reach Codex as the upstream writes them, on the
// translated path its web_search sends a request down (echo_ts on Discord:
// on Copilot, Codex saw nothing for the minutes the model wrote a patch or
// a command, then all of it at once). A function's arguments come as
// function_call_arguments deltas, a custom tool's input (apply_patch) as
// custom_tool_call_input deltas, each before the upstream's call is done;
// the finished items are what they were.
func TestCallArgsStreamWhenTranslated(t *testing.T) {
	const patch = "*** Begin Patch\n*** Add File: a.go\n+package a\n+// \"quoted\" \\ é 😀\n*** End Patch\n"
	whole, _ := json.Marshal(map[string]string{"input": patch})
	split := 24 // inside the patch: {"input":"*** Begin Pat
	for _, c := range []struct {
		name, fn     string
		first, rest  string
		delta        string // the event the client is given the call's text by
		wantFirst    string // what that event's first delta says
		wantItemType string
	}{
		{"function", "exec_command", `{"cmd":"go test ./...",`, `"yield_time_ms":1000}`,
			"response.function_call_arguments.delta", `{"cmd":"go test ./...",`, "function_call"},
		{"custom", "apply_patch", string(whole[:split]), string(whole[split:]),
			"response.custom_tool_call_input.delta", patch[:split-len(`{"input":"`)], "custom_tool_call"},
	} {
		t.Run(c.name, func(t *testing.T) {
			fresh(t)
			up := &gatedCall{name: c.fn, first: c.first, rest: c.rest, seen: make(chan struct{}), wrote: make(chan struct{})}
			vendor := httptest.NewServer(up)
			t.Cleanup(vendor.Close)
			if err := provider.Save(provider.Provider{ID: "rel", Name: "Rel", Key: "k", Responses: vendor.URL + "/v1", Models: []string{"m"}}); err != nil {
				t.Fatal(err)
			}
			srv := httptest.NewServer(New().Handler())
			t.Cleanup(srv.Close)
			body := `{"model":"rel/m","stream":true,"input":[{"role":"user","content":"add a.go"}],
				"tools":[{"type":"function","name":"exec_command","description":"Runs a command.","strict":false,
				  "parameters":{"type":"object","properties":{"cmd":{"type":"string"},"yield_time_ms":{"type":"number"}},"required":["cmd"]}},
				 {"type":"custom","name":"apply_patch","description":"Edits files.","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}},
				 {"type":"web_search"}],"tool_choice":"auto"}`
			res, err := http.Post(srv.URL+"/v1/responses", "application/json", strings.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			if res.StatusCode != 200 {
				b, _ := io.ReadAll(res.Body)
				t.Fatalf("%d %s", res.StatusCode, b)
			}
			var deltas []string
			var done map[string]any
			early := false
			sc := bufio.NewScanner(res.Body)
			sc.Buffer(make([]byte, 1<<20), 1<<20)
			for sc.Scan() {
				data, ok := strings.CutPrefix(sc.Text(), "data: ")
				if !ok {
					continue
				}
				var ev struct {
					Type  string         `json:"type"`
					Delta string         `json:"delta"`
					Item  map[string]any `json:"item"`
				}
				if json.Unmarshal([]byte(data), &ev) != nil {
					continue
				}
				switch {
				case ev.Type == c.delta:
					if len(deltas) == 0 {
						select {
						case <-up.wrote:
						default:
							early = true // the upstream is still holding the rest
						}
						close(up.seen)
					}
					deltas = append(deltas, ev.Delta)
				case ev.Type == "response.output_item.done" && ev.Item["type"] == c.wantItemType:
					done = ev.Item
				case ev.Type == "response.output_item.done" && done == nil:
					// another kind of item: what the call came back as
					done = ev.Item
				}
			}
			if !early {
				t.Fatalf("no %s before the upstream's call was done: deltas %q", c.delta, deltas)
			}
			if deltas[0] != c.wantFirst {
				t.Errorf("first delta %q, want %q", deltas[0], c.wantFirst)
			}
			if done == nil || done["type"] != c.wantItemType {
				t.Fatalf("finished item %v", done)
			}
			if c.wantItemType == "custom_tool_call" {
				if done["input"] != patch || strings.Join(deltas, "") != patch {
					t.Errorf("input %q, deltas %q, want %q", done["input"], strings.Join(deltas, ""), patch)
				}
			} else if done["arguments"] != c.first+c.rest || strings.Join(deltas, "") != c.first+c.rest {
				t.Errorf("arguments %q, deltas %q", done["arguments"], strings.Join(deltas, ""))
			}
		})
	}
}

// A custom tool's input read out of its arguments a piece at a time is
// the input the whole arguments hold, wherever the pieces are cut: in an
// escape, between a surrogate pair's halves, inside a character.
func TestCustomInputStreams(t *testing.T) {
	patch := "*** Begin Patch\n+\"q\" \\ é 😀 <a>&\t\n*** End Patch\n"
	marshalled, _ := json.Marshal(map[string]string{"input": patch})
	for _, c := range []struct{ args, want string }{
		{string(marshalled), patch},
		{`{"input":"\ud83d\ude00 \u00e9\u0041\/\n"}`, "😀 éA/\n"},
		{`{"path":"a.go","input":"x\"y"}`, `x"y`},
		{`{"meta":{"input":"no","list":["input"]},"input":"yes"}`, "yes"},
		{`{ "input" : "spaced" }`, "spaced"},
		{`"a lone string"`, "a lone string"},
		{`{"input":"lone \ud83d half"}`, "lone \ufffd half"},
		{`{"input":5}`, ""},
		{`{"other":"x"}`, ""},
		{`not json`, ""},
		// what a model offered the tool as a function writes that JSON
		// can't read, read as it meant it (Codex in code mode saw exec stop
		// at each of these while the model went on): an escape JSON hasn't
		{`{"input":"rg -n -o -I '\d+' notes.txt"}`, `rg -n -o -I '\d+' notes.txt`},
		{`{"input":"Get-ChildItem -Filter \"*.md\" C:\Users\a\Desktop"}`, `Get-ChildItem -Filter "*.md" C:\Users\a\Desktop`},
		{`{"input":"\x41 \u12 \uZZZZ"}`, `\x41 \u12 \uZZZZ`},
		{`{"input":"a\\\d \ud83d\ude00\q"}`, `a\\d 😀\q`},
		// a raw control character
		{"{\"input\":\"tab\there\nnew line\"}", "tab\there\nnew line"},
		// a quote it didn't escape, where the arguments don't go on from it
		{`{"input":"q({search_query:"go generics"})"}`, `q({search_query:"go generics"})`},
		{`{"input":"print("a", "b")"}`, `print("a", "b")`},
		{`{"input":"x = {"k": "v"}; y"}`, `x = {"k": "v"}; y`},
		{`{"input":"say "hi"}`, `say "hi`},
		{`"lone "quoted" \d"`, `lone "quoted" \d`},
		// a quote the arguments go on from still ends it
		{`{"input":"a", "path":"b.go"}`, "a"},
		{`{"input":"a" , "p\"q" : 1}`, "a"},
		{"{\"input\":\"a\" }\n", "a"},
		// arguments cut off: an escape, a character, a pair's half at the end
		{`{"input":"cut \`, `cut \`},
		{`{"input":"cut \u00`, `cut \u00`},
		{`{"input":"cut \ud83d`, "cut \ufffd"},
		{"{\"input\":\"cut \xe2\x82", "cut \ufffd\ufffd"},
	} {
		for _, size := range []int{1, 2, 3, 5, 7, len(c.args)} {
			var s customStream
			var got strings.Builder
			for i := 0; i < len(c.args); i += size {
				got.WriteString(s.feed(c.args[i:min(i+size, len(c.args))]))
			}
			got.WriteString(s.finish())
			if got.String() != c.want {
				t.Errorf("%s in pieces of %d: %q, want %q", c.args, size, got.String(), c.want)
			}
			if c.want != "" && customInput(c.args) != c.want {
				t.Errorf("%s: customInput %q", c.args, customInput(c.args))
			}
		}
	}
}

// What streams before the arguments end is given as it comes: a quote
// waits only on what follows it, and the text before it isn't held back.
func TestCustomInputStreamsAhead(t *testing.T) {
	var s customStream
	if got := s.feed(`{"input":"rg '\d`); got != `rg '\d` {
		t.Fatalf("before the quote's end is known: %q", got)
	}
	if got := s.feed(`+' "x`); got != `+' "x` {
		t.Fatalf("a quote followed by more input: %q", got)
	}
	if got := s.feed(`"`); got != "" {
		t.Fatalf("a quote at the pieces' end: %q", got)
	}
	if got := s.feed(`}`); got != "" {
		t.Fatalf("a quote then the object's end: %q", got)
	}
	if got := s.finish(); got != "" {
		t.Fatalf("the end: %q", got)
	}
	if got := customInput(`{"input":"rg '\d+' "x"}`); got != `rg '\d+' "x` {
		t.Fatalf("customInput %q", got)
	}
}

// A Responses upstream's argument deltas are given as they come, except
// those that make the arguments whole JSON, which wait for the call's done
// events: what was given and what follows add up to the fuller arguments.
func TestResponsesCallArgsStream(t *testing.T) {
	var d responsesDecoder
	var got []string
	emit := func(ev Event) {
		if ev.Kind == KToolArgs {
			got = append(got, ev.Text)
		}
	}
	d.decode(`{"type":"response.output_item.added","item":{"type":"function_call","call_id":"c1","name":"f"}}`, emit)
	for _, x := range []string{`{"cmd":"a}`, `\"]{",`, `"n":[1,{}]`} {
		q, _ := json.Marshal(x)
		d.decode(`{"type":"response.function_call_arguments.delta","delta":`+string(q)+`}`, emit)
	}
	if want := []string{`{"cmd":"a}`, `\"]{",`, `"n":[1,{}]`}; strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("before the end %q, want %q", got, want)
	}
	d.decode(`{"type":"response.function_call_arguments.delta","delta":"}"}`, emit)
	if len(got) != 3 {
		t.Fatalf("the delta that makes it whole was given before the call's done: %q", got)
	}
	d.decode(`{"type":"response.output_item.done","item":{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"cmd\":\"a}\\\"]{\",\"n\":[1,{}],\"more\":true}"}}`, emit)
	if all := strings.Join(got, ""); all != `{"cmd":"a}\"]{","n":[1,{}],"more":true}` {
		t.Fatalf("arguments %s", all)
	}
}

// A custom tool's input a model wrote that JSON can't read reaches Codex
// as it streams, and the deltas add up to the finished item's input: the
// translated exec in code mode stopped at a regex's \d, a quote it hadn't
// escaped, and Codex ran the whole arguments as its input. What waited on
// the arguments' end (a backslash they end with) is given before the
// input's done.
func TestCustomInputThatIsntJSONStreams(t *testing.T) {
	for _, c := range []struct{ args, want string }{
		{"{\"input\":\"rg -n '\\d+' \"a b\"\tc\"}", "rg -n '\\d+' \"a b\"\tc"},
		{`{"input":"dir C:\Users\a \`, `dir C:\Users\a \`},
	} {
		fresh(t)
		vendor := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "text/event-stream")
			q := func(s string) string { b, _ := json.Marshal(s); return string(b) }
			lines := []string{`data: {"type":"response.created","response":{"id":"r1","model":"m"}}`,
				`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"apply_patch","arguments":""}}`}
			for i := 0; i < len(c.args); i += 3 {
				lines = append(lines, `data: {"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":`+q(c.args[i:min(i+3, len(c.args))])+`}`)
			}
			call := `{"type":"function_call","id":"fc_1","call_id":"call_1","name":"apply_patch","arguments":` + q(c.args) + `,"status":"completed"}`
			lines = append(lines, `data: {"type":"response.output_item.done","output_index":0,"item":`+call+`}`,
				`data: {"type":"response.completed","response":{"id":"r1","status":"completed","output":[`+call+`],"usage":{"input_tokens":5,"output_tokens":5}}}`)
			io.WriteString(w, sse(lines...))
		}))
		t.Cleanup(vendor.Close)
		if err := provider.Save(provider.Provider{ID: "rel", Name: "Rel", Key: "k", Responses: vendor.URL + "/v1", Models: []string{"m"}}); err != nil {
			t.Fatal(err)
		}
		body := `{"model":"rel/m","stream":true,"input":[{"role":"user","content":"search"}],
			"tools":[{"type":"custom","name":"apply_patch","description":"Edits files.","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}},
			 {"type":"web_search"}],"tool_choice":"auto"}`
		rec := httptest.NewRecorder()
		New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))
		if rec.Code != 200 {
			t.Fatalf("%d %s", rec.Code, rec.Body)
		}
		var deltas strings.Builder
		var done, item any
		for _, e := range events(rec.Body.String()) {
			switch e["type"] {
			case "response.custom_tool_call_input.delta":
				d, _ := e["delta"].(string)
				deltas.WriteString(d)
			case "response.custom_tool_call_input.done":
				done = e["input"]
			case "response.output_item.done":
				if it, _ := e["item"].(map[string]any); it["type"] == "custom_tool_call" {
					item = it["input"]
				}
			}
		}
		if deltas.String() != c.want || done != c.want || item != c.want {
			t.Errorf("%s: deltas %q, done %q, item %q, want %q", c.args, deltas.String(), done, item, c.want)
		}
	}
}
