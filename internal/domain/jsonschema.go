package domain

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// This file implements Track B's schema-constrained extraction support: a
// draft-07 *subset* JSON Schema parser/validator. The subset is deliberately
// small — it covers the shapes narrow extraction tasks need (objects, arrays,
// scalars, enums, const, required/additionalProperties) so import-time
// validation can reject examples whose `output` does not satisfy the task's
// schema. GBNF grammar generation (for constrained decoding) lives in
// gbnf.go and uses the same parser.
//
// Unsupported keywords are ignored rather than rejected, so schemas authored
// against a fuller draft-07 still load; only structurally-broken schemas
// (non-object root, bad `type`, malformed JSON) are rejected.

// SchemaError describes a JSON Schema problem (authoring or instance).
type SchemaError struct {
	// Path is the JSON pointer-ish location of the offending value ("" for
	// the schema itself).
	Path string `json:"path,omitempty"`
	Msg  string `json:"msg"`
}

func (e *SchemaError) Error() string {
	if e.Path == "" {
		return "json schema: " + e.Msg
	}

	return fmt.Sprintf("json schema at %s: %s", e.Path, e.Msg)
}

// jsonSchema is the parsed subset of a draft-07 schema.
type jsonSchema struct {
	Type                 string
	Properties           map[string]*jsonSchema
	PropertyOrder        []string
	Required             []string
	Items                *jsonSchema
	Enum                 []json.RawMessage
	Const                *json.RawMessage
	AdditionalProperties *bool
	Minimum              *float64
	Maximum              *float64
	MinLength            *int
	MaxLength            *int
	MinItems             *int
	MaxItems             *int
	Pattern              string
	Description          string
}

// allowedSchemaTypes is the set of JSON types this subset understands.
var allowedSchemaTypes = map[string]bool{
	"object": true, "array": true, "string": true,
	"number": true, "integer": true, "boolean": true, "null": true,
}

// ValidateJSONSchema parses and structurally validates a JSON Schema string.
func ValidateJSONSchema(schema string) error {
	schema = strings.TrimSpace(schema)
	if schema == "" {
		return nil // empty schema = "no constraint".
	}

	_, err := parseJSONSchema(schema, "")

	return err
}

// parseJSONSchema parses one schema node, validating the subset's structural
// rules. path is used for error messages.
func parseJSONSchema(raw string, path string) (*jsonSchema, error) {
	var obj map[string]json.RawMessage

	err := json.Unmarshal([]byte(raw), &obj)
	if err != nil {
		return nil, &SchemaError{Path: path, Msg: "schema must be a JSON object"}
	}

	s := &jsonSchema{Properties: map[string]*jsonSchema{}}

	if t, ok := obj["type"]; ok {
		var typeStr string
		if json.Unmarshal(t, &typeStr) != nil {
			// A type may be an array (union); take the first supported entry.
			var typeList []string
			if json.Unmarshal(t, &typeList) == nil {
				for _, cand := range typeList {
					if allowedSchemaTypes[cand] {
						typeStr = cand
						break
					}
				}
			}
		}

		if !allowedSchemaTypes[typeStr] {
			return nil, &SchemaError{Path: path, Msg: fmt.Sprintf("unsupported or missing type %q", typeStr)}
		}

		s.Type = typeStr
	}

	_ = json.Unmarshal(obj["description"], &s.Description)
	_ = json.Unmarshal(obj["required"], &s.Required)
	_ = json.Unmarshal(obj["enum"], &s.Enum)
	_ = json.Unmarshal(obj["pattern"], &s.Pattern)

	if c, ok := obj["const"]; ok {
		cc := c
		s.Const = &cc
	}

	if ap, ok := obj["additionalProperties"]; ok {
		var b bool
		if json.Unmarshal(ap, &b) == nil {
			s.AdditionalProperties = &b
		}
	}

	if props, ok := obj["properties"]; ok {
		var propMap map[string]json.RawMessage
		if json.Unmarshal(props, &propMap) != nil {
			return nil, &SchemaError{Path: path, Msg: "properties must be an object"}
		}

		for name, propRaw := range propMap {
			child, err := parseJSONSchema(string(propRaw), path+"/properties/"+name)
			if err != nil {
				return nil, err
			}

			s.Properties[name] = child
		}

		s.PropertyOrder = sortedKeys(propMap)
	}

	if items, ok := obj["items"]; ok {
		child, err := parseJSONSchema(string(items), path+"/items")
		if err != nil {
			return nil, err
		}

		s.Items = child
	}

	s.Minimum = rawFloat(obj["minimum"])
	s.Maximum = rawFloat(obj["maximum"])
	s.MinLength = rawInt(obj["minLength"])
	s.MaxLength = rawInt(obj["maxLength"])
	s.MinItems = rawInt(obj["minItems"])
	s.MaxItems = rawInt(obj["maxItems"])

	return s, nil
}

// ValidateJSONAgainstSchema validates a JSON document against a schema string.
// An empty schema accepts any valid JSON document.
func ValidateJSONAgainstSchema(schema, doc string) error {
	schema = strings.TrimSpace(schema)
	if schema == "" {
		var anyDoc interface{}
		if json.Unmarshal([]byte(doc), &anyDoc) != nil {
			return &SchemaError{Msg: "output is not valid JSON"}
		}

		return nil
	}

	s, err := parseJSONSchema(schema, "")
	if err != nil {
		return err
	}

	var value interface{}
	if json.Unmarshal([]byte(doc), &value) != nil {
		return &SchemaError{Msg: "output is not valid JSON"}
	}

	return validateValue(s, value, "")
}

// validateValue checks an unmarshalled JSON value against a schema node.
func validateValue(s *jsonSchema, v interface{}, path string) error {
	if s.Const != nil {
		var constVal interface{}

		_ = json.Unmarshal(*s.Const, &constVal)

		if !jsonEqual(constVal, v) {
			return &SchemaError{Path: path, Msg: "value does not equal const"}
		}
	}

	if len(s.Enum) > 0 {
		matched := false

		for _, e := range s.Enum {
			var ev interface{}

			_ = json.Unmarshal(e, &ev)

			if jsonEqual(ev, v) {
				matched = true
				break
			}
		}

		if !matched {
			return &SchemaError{Path: path, Msg: "value not in enum"}
		}
	}

	if s.Type != "" && !typeMatches(s.Type, v) {
		return &SchemaError{Path: path, Msg: "expected type " + s.Type}
	}

	switch val := v.(type) {
	case string:
		return validateString(s, val, path)
	case float64:
		return validateNumber(s, val, path)
	case []interface{}:
		return validateArray(s, val, path)
	case map[string]interface{}:
		return validateObject(s, val, path)
	default:
		return nil
	}
}

func validateString(s *jsonSchema, val, path string) error {
	if s.MinLength != nil && len([]rune(val)) < *s.MinLength {
		return &SchemaError{Path: path, Msg: "string shorter than minLength"}
	}

	if s.MaxLength != nil && len([]rune(val)) > *s.MaxLength {
		return &SchemaError{Path: path, Msg: "string longer than maxLength"}
	}

	if s.Pattern != "" {
		if re, err := regexp.Compile(s.Pattern); err == nil && !re.MatchString(val) {
			return &SchemaError{Path: path, Msg: "string does not match pattern"}
		}
	}

	return nil
}

func validateNumber(s *jsonSchema, val float64, path string) error {
	if s.Minimum != nil && val < *s.Minimum {
		return &SchemaError{Path: path, Msg: "number below minimum"}
	}

	if s.Maximum != nil && val > *s.Maximum {
		return &SchemaError{Path: path, Msg: "number above maximum"}
	}

	return nil
}

func validateArray(s *jsonSchema, val []interface{}, path string) error {
	if s.MinItems != nil && len(val) < *s.MinItems {
		return &SchemaError{Path: path, Msg: "array shorter than minItems"}
	}

	if s.MaxItems != nil && len(val) > *s.MaxItems {
		return &SchemaError{Path: path, Msg: "array longer than maxItems"}
	}

	if s.Items != nil {
		for i, item := range val {
			err := validateValue(s.Items, item, fmt.Sprintf("%s/%d", path, i))
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func validateObject(s *jsonSchema, val map[string]interface{}, path string) error {
	for _, req := range s.Required {
		if _, ok := val[req]; !ok {
			return &SchemaError{Path: path, Msg: "missing required property " + strconv.Quote(req)}
		}
	}

	for _, name := range sortedKeysInterface(val) {
		prop, known := s.Properties[name]
		if known {
			err := validateValue(prop, val[name], path+"/"+name)
			if err != nil {
				return err
			}

			continue
		}

		if s.AdditionalProperties != nil && !*s.AdditionalProperties {
			return &SchemaError{Path: path, Msg: "unexpected additional property " + strconv.Quote(name)}
		}
	}

	return nil
}

// typeMatches reports whether a decoded JSON value satisfies a schema type.
func typeMatches(t string, v interface{}) bool {
	switch t {
	case "object":
		_, ok := v.(map[string]interface{})
		return ok
	case "array":
		_, ok := v.([]interface{})
		return ok
	case "string":
		_, ok := v.(string)
		return ok
	case "number":
		_, ok := v.(float64)
		return ok
	case "integer":
		f, ok := v.(float64)
		return ok && f == math.Trunc(f)
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "null":
		return v == nil
	default:
		return true
	}
}

// --- small helpers ---------------------------------------------------------.

func rawFloat(r json.RawMessage) *float64 {
	if len(r) == 0 {
		return nil
	}

	var f float64
	if json.Unmarshal(r, &f) != nil {
		return nil
	}

	return &f
}

func rawInt(r json.RawMessage) *int {
	if len(r) == 0 {
		return nil
	}

	var i int
	if json.Unmarshal(r, &i) != nil {
		return nil
	}

	return &i
}

func sortedKeys(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}

	sort.Strings(out)

	return out
}

func sortedKeysInterface(m map[string]interface{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}

	sort.Strings(out)

	return out
}

// jsonEqual compares two decoded JSON values for deep equality (numbers
// compared numerically).
func jsonEqual(a, b interface{}) bool {
	an, aok := a.(float64)
	bn, bok := b.(float64)

	if aok && bok {
		return an == bn
	}

	switch av := a.(type) {
	case map[string]interface{}:
		bv, ok := b.(map[string]interface{})
		if !ok || len(av) != len(bv) {
			return false
		}

		for k, v := range av {
			if !jsonEqual(v, bv[k]) {
				return false
			}
		}

		return true
	case []interface{}:
		bv, ok := b.([]interface{})
		if !ok || len(av) != len(bv) {
			return false
		}

		for i := range av {
			if !jsonEqual(av[i], bv[i]) {
				return false
			}
		}

		return true
	default:
		return a == b
	}
}
