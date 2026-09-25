package domain

import (
	"fmt"
	"strconv"
	"strings"
)

// GenerateGBNF builds a llama.cpp GBNF grammar from a JSON Schema so
// llama-server (via its `grammar` parameter) constrains decoding to
// schema-valid JSON. An empty or unparseable schema yields "" (no grammar
// constraint). This is Track B's mechanism for guaranteeing schema-valid
// JSON on 100% of served outputs.
func GenerateGBNF(schema string) string {
	schema = strings.TrimSpace(schema)
	if schema == "" {
		return ""
	}

	s, err := parseJSONSchema(schema, "")
	if err != nil {
		return ""
	}

	g := &gbnfGen{rules: map[string]string{}}
	start := g.emit(s)

	var sb strings.Builder

	sb.WriteString("root ::= " + start + "\n")

	for _, name := range g.order {
		sb.WriteString(name + " ::= " + g.rules[name] + "\n")
	}

	return sb.String()
}

// gbnfGen accumulates GBNF rules with stable numbering.
type gbnfGen struct {
	rules map[string]string
	order []string
	seq   int
}

func (g *gbnfGen) fresh(prefix string) string {
	g.seq++
	return fmt.Sprintf("%s%d", prefix, g.seq)
}

func (g *gbnfGen) define(name, body string) string {
	if _, exists := g.rules[name]; exists {
		return name
	}

	g.rules[name] = body
	g.order = append(g.order, name)

	return name
}

// emit returns a GBNF reference (a rule name) matching the schema node.
func (g *gbnfGen) emit(s *jsonSchema) string {
	switch s.Type {
	case "object":
		return g.emitObject(s)
	case "array":
		return g.emitArray(s)
	case "string":
		return g.emitString(s)
	case "number":
		return g.define(g.fresh("num"), `"-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?`)
	case "integer":
		return g.define(g.fresh("int"), `"-"? ("0" | [1-9] [0-9]*)`)
	case "boolean":
		return g.define(g.fresh("bool"), `"true" | "false"`)
	case "null":
		return g.define(g.fresh("null"), `"null"`)
	default:
		if len(s.Enum) > 0 || s.Const != nil {
			return g.emitEnumOrConst(s)
		}

		return g.emitAny()
	}
}

func (g *gbnfGen) emitEnumOrConst(s *jsonSchema) string {
	var variants []string

	if s.Const != nil {
		variants = append(variants, strconv.Quote(string(*s.Const)))
	}

	for _, e := range s.Enum {
		variants = append(variants, strconv.Quote(string(e)))
	}

	if len(variants) == 0 {
		return g.emitAny()
	}

	return g.define(g.fresh("enum"), strings.Join(variants, " | "))
}

func (g *gbnfGen) emitString(s *jsonSchema) string {
	if len(s.Enum) > 0 || s.Const != nil {
		return g.emitEnumOrConst(s)
	}

	// A JSON string: opening quote, then non-quote/escape chars or escapes.
	return g.define(g.fresh("str"), `"\"" ([^"\\] | "\\" ["\\/bfnrt] | "\\u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])* "\""`)
}

func (g *gbnfGen) emitArray(s *jsonSchema) string {
	item := g.emitAny()
	if s.Items != nil {
		item = g.emit(s.Items)
	}

	name := g.fresh("arr")

	return g.define(name, `"[" ws (`+item+` (ws "," ws `+item+`)*)? ws "]"`)
}

func (g *gbnfGen) emitObject(s *jsonSchema) string {
	if len(s.Properties) == 0 {
		return g.define(g.fresh("obj"), `"{" ws "}"`)
	}

	// Resolve each property's value grammar, partitioned into required and
	// optional groups (matching JSON Schema's required keyword).
	requiredSet := map[string]bool{}
	for _, r := range s.Required {
		requiredSet[r] = true
	}

	var requiredParts, optionalParts []string

	for _, name := range s.PropertyOrder {
		valRef := g.emit(s.Properties[name])
		part := `"\"" ` + strconv.Quote(name) + ` "\"" ws ":" ws ` + valRef

		if requiredSet[name] {
			requiredParts = append(requiredParts, part)
		} else {
			optionalParts = append(optionalParts, part)
		}
	}

	var body strings.Builder
	body.WriteString(`"{" ws `)

	switch {
	case len(requiredParts) > 0 && len(optionalParts) > 0:
		body.WriteString(strings.Join(requiredParts, ` ws "," ws `))
		body.WriteString(` (ws "," ws (` + strings.Join(optionalParts, ` ws "," ws `) + `))?`)
	case len(requiredParts) > 0:
		body.WriteString(strings.Join(requiredParts, ` ws "," ws `))
	case len(optionalParts) > 0:
		body.WriteString(`(` + strings.Join(optionalParts, ` | `) + `)?`)
	}

	body.WriteString(` ws "}"`)

	return g.define(g.fresh("obj"), body.String())
}

// emitAny defines (once) and returns the rule matching any JSON value.
func (g *gbnfGen) emitAny() string {
	if _, ok := g.rules["anyjson"]; ok {
		return "anyjson"
	}

	// Define the shared primitive rules first so anyjson can reference them.
	g.define("ws", `[ \t\n]*`)
	g.define("any-object", `"{" ws "}"`)
	g.define("any-array", `"[" ws "]"`)
	g.define("any-string", `"\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""`)
	g.define("any-number", `"-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?`)
	g.define("any-boolean", `"true" | "false"`)
	g.define("any-null", `"null"`)
	g.define("anyjson", `any-object | any-array | any-string | any-number | any-boolean | any-null`)

	return "anyjson"
}
