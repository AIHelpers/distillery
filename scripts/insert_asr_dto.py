import io
import sys

MARKER = "// addASRAudioExampleRequest is the body of POST /tasks/{id}/examples/asr."

DTO = """// addASRAudioExampleRequest is the body of POST /tasks/{id}/examples/asr.
// Audio is base64-encoded (data-URL prefixes like "data:audio/wav;base64,"
// are stripped by the handler) so a single-clip add works as plain JSON
// without a multipart round trip.
type addASRAudioExampleRequest struct {
\tAudioBase64 string `json:"audio_base64"`
\tFilename    string `json:"filename,omitempty"`
\tText        string `json:"text"`
\tSpeaker     string `json:"speaker,omitempty"`
}

// transcribeRequest is the body of POST /inference/{id}/transcribe.
// Audio is base64-encoded (data-URL prefixes are stripped).
type transcribeRequest struct {
\tAudioBase64 string `json:"audio_base64"`
\tFilename    string `json:"filename,omitempty"`
\t// Language optionally pins the transcription language (ISO 639-1);
\t// empty lets the model auto-detect.
\tLanguage string `json:"language,omitempty"`
}

"""


def insert_before(path, anchor, text):
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        content = f.read()
    if MARKER in content:
        print("SKIP: already inserted")
        return
    if anchor not in content:
        print("ANCHOR NOT FOUND in", path)
        sys.exit(1)
    content = content.replace(anchor, text + anchor, 1)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print("INSERTED into", path)


if __name__ == "__main__":
    insert_before(
        "internal/delivery/http/dto.go",
        "// addVisionExampleRequest is the body of POST /tasks/{id}/examples/vision.",
        DTO,
    )