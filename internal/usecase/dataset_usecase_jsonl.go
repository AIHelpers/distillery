package usecase

import (
	"bufio"
	"encoding/json"
	"strings"

	"distillery/internal/domain"
)

// JSONLFormat identifies a fine-tuning JSONL dialect. Empty means auto-detect
// from the first record.
type JSONLFormat string

const (
	JSONLAlpaca JSONLFormat = "alpaca" // {"instruction","input","output",...}.
	JSONLChat   JSONLFormat = "chat"   // {"messages":[{"role","content"},...],...}.
)

// ImportJSONL parses Alpaca-style or chat-style JSONL content and loads each
// record as an example, then re-runs curation.
//
// Alpaca records map to:
//
//	input  = instruction (+ "\n\n" + input when input is present)
//	output = output
//
// Chat records map to:
//
//	input  = concatenation of every non-assistant message's content
//	output = the last assistant message's content
//
// An empty format hint auto-detects from the first non-empty line.
func (u *DatasetUsecase) ImportJSONL(taskID, content, formatHint string) (*domain.DatasetStats, error) {
	_, err := u.tasks.Get(taskID)
	if err != nil {
		return nil, err
	}

	pairs, err := parseJSONL(content, strings.ToLower(strings.TrimSpace(formatHint)))
	if err != nil {
		return nil, err
	}

	if len(pairs) == 0 {
		return nil, domain.ErrInvalidInput
	}

	return u.AddExamples(taskID, pairs)
}

// maxJSONLLine is the largest single JSONL record we'll buffer (10 MiB),
// generous enough for long chat transcripts.
const maxJSONLLine = 10 << 20

func parseJSONL(content, formatHint string) ([]ExamplePair, error) {
	var (
		pairs    []ExamplePair
		detected = formatHint
	)

	scanner := bufio.NewScanner(strings.NewReader(content))
	scanner.Buffer(make([]byte, 128*1024), maxJSONLLine)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Auto-detect from the first non-empty record when no hint given.
		if detected == "" {
			detected = detectJSONLFormat(line)
			if detected == "" {
				return nil, domain.ErrInvalidInput
			}
		}

		var (
			pair ExamplePair
			ok   bool
		)

		switch JSONLFormat(detected) {
		case JSONLAlpaca:
			pair, ok = parseAlpacaLine(line)
		case JSONLChat:
			pair, ok = parseChatLine(line)
		default:
			return nil, domain.ErrInvalidInput
		}

		if ok {
			pairs = append(pairs, pair)
		}
	}

	err := scanner.Err()
	if err != nil {
		return nil, domain.ErrInvalidInput
	}

	return pairs, nil
}

// detectJSONLFormat inspects a single JSON object to decide whether it is an
// Alpaca record (has "instruction") or a chat record (has "messages").
func detectJSONLFormat(line string) string {
	var raw map[string]json.RawMessage

	err := json.Unmarshal([]byte(line), &raw)
	if err != nil {
		return ""
	}

	if _, ok := raw["messages"]; ok {
		return string(JSONLChat)
	}

	if _, ok := raw["instruction"]; ok {
		return string(JSONLAlpaca)
	}

	return ""
}

// parseAlpacaLine converts one Alpaca-style record into an ExamplePair.
// Records without a non-empty instruction+input or output are skipped.
func parseAlpacaLine(line string) (ExamplePair, bool) {
	var rec struct {
		Instruction string `json:"instruction"`
		Input       string `json:"input"`
		Output      string `json:"output"`
	}

	err := json.Unmarshal([]byte(line), &rec)
	if err != nil {
		return ExamplePair{}, false
	}

	instruction := strings.TrimSpace(rec.Instruction)
	extraInput := strings.TrimSpace(rec.Input)
	output := strings.TrimSpace(rec.Output)

	if output == "" {
		return ExamplePair{}, false
	}

	fullInput := instruction
	if extraInput != "" {
		if fullInput != "" {
			fullInput += "\n\n"
		}

		fullInput += extraInput
	}

	if strings.TrimSpace(fullInput) == "" {
		return ExamplePair{}, false
	}

	return ExamplePair{Input: strings.TrimSpace(fullInput), Output: output}, true
}

// chatMessage mirrors the OpenAI-style {"role","content"} shape used by
// chat-template JSONL datasets.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// parseChatLine converts one chat-style record into an ExamplePair.
// The last assistant message becomes the output; all other messages
// (system, user, tool, …) are concatenated into the input.
func parseChatLine(line string) (ExamplePair, bool) {
	var rec struct {
		Messages []chatMessage `json:"messages"`
	}

	err := json.Unmarshal([]byte(line), &rec)
	if err != nil {
		return ExamplePair{}, false
	}

	if len(rec.Messages) == 0 {
		return ExamplePair{}, false
	}

	var promptParts []string

	output := ""

	for _, msg := range rec.Messages {
		content := strings.TrimSpace(msg.Content)
		if content == "" {
			continue
		}

		switch strings.ToLower(msg.Role) {
		case "assistant":
			output = content // last assistant message wins.
		default:
			promptParts = append(promptParts, content)
		}
	}

	input := strings.TrimSpace(strings.Join(promptParts, "\n\n"))
	output = strings.TrimSpace(output)

	if input == "" || output == "" {
		return ExamplePair{}, false
	}

	return ExamplePair{Input: input, Output: output}, true
}
