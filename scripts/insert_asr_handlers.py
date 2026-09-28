import io
import sys

PATH = "internal/delivery/http/dataset_handler.go"

HANDLERS = '''// AddASRAudioExample POST /api/v1/tasks/{taskID}/examples/asr adds one
// audio+transcript example. The audio is base64-encoded in the JSON body
// (data-URL prefixes like "data:audio/wav;base64," are stripped).
func (h *DatasetHandler) AddASRAudioExample(w http.ResponseWriter, r *http.Request, taskID string) {
\tvar req addASRAudioExampleRequest

\terr := decodeJSON(r, &req)
\tif err != nil {
\t\twriteError(w, http.StatusBadRequest, "invalid JSON body")
\t\treturn
\t}

\tif strings.TrimSpace(req.Text) == "" {
\t\twriteError(w, http.StatusBadRequest, "text (transcript) is required")
\t\treturn
\t}

\taudioData, err := decodeImageBase64(req.AudioBase64)
\tif err != nil {
\t\twriteError(w, http.StatusBadRequest, "audio_base64 is not valid base64 audio data")
\t\treturn
\t}

\tfilename := strings.TrimSpace(req.Filename)
\tif filename == "" {
\t\tfilename = "clip.wav"
\t}

\tstats, err := h.uc.AddASRAudioExample(taskID, audioData, filename, req.Text, req.Speaker)
\tif err != nil {
\t\thandleErr(w, err)
\t\treturn
\t}

\twriteJSON(w, http.StatusOK, stats)
}

// ImportASRZIP POST /api/v1/tasks/{taskID}/examples/import-asr-zip
// bulk-loads a ZIP archive of audio files plus a manifest.jsonl (see
// docs/speech-asr.md for the manifest shape).
func (h *DatasetHandler) ImportASRZIP(w http.ResponseWriter, r *http.Request, taskID string) {
\tbody, err := io.ReadAll(io.LimitReader(r.Body, 500<<20)) // 500MB cap.
\tif err != nil {
\t\twriteError(w, http.StatusBadRequest, "failed to read request body")
\t\treturn
\t}

\tstats, err := h.uc.ImportASRZIP(taskID, body)
\tif err != nil {
\t\thandleErr(w, err)
\t\treturn
\t}

\twriteJSON(w, http.StatusOK, stats)
}

// ASRStats GET /api/v1/tasks/{taskID}/examples/asr-stats reports the
// audio-side dataset summary (total hours, duration histogram, per-speaker
// counts).
func (h *DatasetHandler) ASRStats(w http.ResponseWriter, _ *http.Request, taskID string) {
\tstats, err := h.uc.ASRDatasetStats(taskID)
\tif err != nil {
\t\thandleErr(w, err)
\t\treturn
\t}

\twriteJSON(w, http.StatusOK, stats)
}

'''

ANCHOR = "// AddVisionExample POST /api/v1/tasks/{taskID}/examples/vision adds one"


def main():
    with io.open(PATH, "r", encoding="utf-8", newline="") as f:
        content = f.read()

    if "AddASRAudioExample" in content:
        print("SKIP: handlers already present")
        return

    if ANCHOR not in content:
        print("ANCHOR NOT FOUND")
        sys.exit(1)

    content = content.replace(ANCHOR, HANDLERS + ANCHOR, 1)

    with io.open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    print("INSERTED handlers into", PATH)


if __name__ == "__main__":
    main()