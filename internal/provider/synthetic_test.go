package provider

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/agentenv"
	"github.com/yetone/magpie/internal/catalog"
)

// Synthetic (dev.synthetic.new/docs/api/overview): OpenAI-compatible chat
// completions and /models at /openai/v1, Anthropic-compatible Messages at
// /anthropic (no Responses endpoint), models.dev's "synthetic" catalog,
// unsaid billing (subscription + usage-based on one host), not balance-probed,
// and recognized from URL and OpenCode imports.
func TestSyntheticPreset(t *testing.T) {
	p, err := FromPreset("synthetic")
	if err != nil {
		t.Fatal(err)
	}
	if p.Name != "Synthetic" || p.Icon != "synthetic" || p.Catalog != "synthetic" || p.Website != "https://dev.synthetic.new/docs/api/overview" || p.KeysURL != "https://synthetic.new/settings/api" {
		t.Fatalf("preset: %+v", p)
	}
	if p.Chat != "https://api.synthetic.new/openai/v1" || p.Anthropic != "https://api.synthetic.new/anthropic" || p.Responses != "" {
		t.Fatalf("endpoints: chat=%q responses=%q anthropic=%q", p.Chat, p.Responses, p.Anthropic)
	}
	if pr := Preset("synthetic"); pr == nil || pr.Kind != KindRelay || pr.NoKey {
		t.Fatalf("preset def: %+v", pr)
	}
	p.Key = "syn_test_key"
	if got := p.BillingKind(); got != "" {
		t.Errorf("BillingKind = %q, want unsaid (same endpoints serve subscription and pay-as-you-go)", got)
	}
	if probesBalance(p) {
		t.Error("Synthetic is balance-probed; it uses /v2/quotas instead")
	}
	if !notProbed("api.synthetic.new") {
		t.Error("api.synthetic.new should be in noProbeHosts")
	}
	// Bundled monochrome SVG icon must exist.
	if b, err := os.ReadFile("../gui/assets/icons/synthetic.svg"); err != nil || !strings.Contains(string(b), "stroke=\"currentColor\"") {
		t.Fatalf("synthetic icon missing or unexpected: %v", err)
	}
	// Importing either documented endpoint resolves to the full preset.
	for _, e := range []endpoints{
		{chat: "https://api.synthetic.new/openai/v1/"},
		{anthropic: "https://api.synthetic.new/anthropic"},
	} {
		im, why := imported("Synthetic", "syn-key", e, nil)
		if why != "" || im.Preset != "synthetic" || im.Icon != "synthetic" || im.Chat != "https://api.synthetic.new/openai/v1" || im.Anthropic != "https://api.synthetic.new/anthropic" {
			t.Fatalf("imported %+v: why=%q p=%+v", e, why, im)
		}
	}
	// OpenCode auth.json stores Synthetic under models.dev id "synthetic".
	dir := t.TempDir()
	path := filepath.Join(dir, "auth.json")
	if err := os.WriteFile(path, []byte(`{"synthetic": {"type": "api", "key": "syn-oc-key"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	ims, err := readOpenCodeAuth(path)
	if err != nil || len(ims) != 1 || ims[0].Skip != "" || ims[0].Provider.Preset != "synthetic" || ims[0].Provider.Key != "syn-oc-key" {
		t.Fatalf("OpenCode import: err=%v ims=%+v", err, ims)
	}
}

// Synthetic's live /openai/v1/models response (testdata/synthetic-models.json)
// puts input_modalities, output_modalities, max_output_length,
// supported_features and reasoning_parameters.efforts at each row's top level.
// Only at api.synthetic.new: another host with the same rows is read as before.
func TestSyntheticLiveList(t *testing.T) {
	body, err := os.ReadFile("testdata/synthetic-models.json")
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/openai/v1/models" {
			rw.WriteHeader(http.StatusNotFound)
			return
		}
		rw.Write(body)
	}))
	defer srv.Close()
	old := http.DefaultClient.Transport
	t.Cleanup(func() { http.DefaultClient.Transport = old })
	http.DefaultClient.Transport = rewrite{srv}

	ms, _, err := catalog.FetchAt(context.Background(), "https://api.synthetic.new/openai/v1", "k", false, nil)
	if err != nil {
		t.Fatal(err)
	}
	by := map[string]catalog.Model{}
	for _, m := range ms {
		by[m.ID] = m
	}
	if len(by) != 12 {
		t.Fatalf("expected 12 models, got %d", len(by))
	}
	if m := by["syn:large:vision"]; m.Context != 524288 || m.Output != 65536 || !m.Images || m.ImageInput == nil || !*m.ImageInput || !m.Reasoning || !slices.Equal(m.Efforts, []string{"low", "high", "max"}) {
		t.Errorf("syn:large:vision: %+v", m)
	}
	if m := by["syn:small:text"]; m.Context != 196608 || m.Output != 65536 || m.Images || m.ImageInput == nil || *m.ImageInput || !m.Reasoning || !slices.Equal(m.Efforts, []string{"none", "low", "medium", "high"}) {
		t.Errorf("syn:small:text: %+v", m)
	}
	if m := by["hf:zai-org/GLM-5.3"]; m.Context != 524288 || m.Output != 65536 || m.Images || !m.Reasoning || !slices.Equal(m.Efforts, []string{"low", "high", "max"}) {
		t.Errorf("hf:zai-org/GLM-5.3: %+v", m)
	}
	if m := by["hf:moonshotai/Kimi-K3.1-Pro"]; m.Name != "Kimi K3.1 Pro" || m.Context != 524288 || m.Output != 65536 || !m.Images {
		t.Errorf("hf:moonshotai/Kimi-K3.1-Pro: %+v", m)
	}

	// Another host's rows are left to standard OpenAI parsing.
	http.DefaultClient.Transport = old
	other, _, err := catalog.FetchAt(context.Background(), srv.URL+"/openai/v1", "k", false, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range other {
		if m.ID == "syn:large:vision" && (m.Output != 0 || len(m.Efforts) != 0 || m.ImageInput != nil) {
			t.Errorf("non-Synthetic host got Synthetic-specific fields: %+v", m)
		}
	}
}

func TestReadSyntheticPlan(t *testing.T) {
	// Source lookup for both Chat and Anthropic bases.
	for _, p := range []Provider{
		{Chat: "https://api.synthetic.new/openai/v1"},
		{Anthropic: "https://api.synthetic.new/anthropic"},
	} {
		src, ok := planQuotaSourceOf(p)
		if !ok || src.url != "https://api.synthetic.new/v2/quotas" || !src.bearer || src.sure {
			t.Fatalf("planQuotaSourceOf(%+v) = %+v, ok=%v", p, src, ok)
		}
	}

	// Current planWeeklyLimit: percentage points (including < 1%), ignores retired subscription counter.
	_, ws, err := readSyntheticPlan([]byte(`{
		"planWeeklyLimit": {"nextRegenAt": "2026-11-22T00:00:00Z", "percentRemaining": 0.5, "nextRegenPercent": 12.5},
		"subscription": {"limit": 135, "requests": 10, "renewsAt": "2026-11-21T10:00:00Z"},
		"search": {"hourly": {"limit": 250, "requests": 25, "renewsAt": "2026-11-21T08:00:00Z"}}
	}`))
	if err != nil || len(ws) != 2 {
		t.Fatalf("planWeeklyLimit: err=%v ws=%+v", err, ws)
	}
	if w := ws[0]; w.Name != "7 days" || w.Span != 7*24*time.Hour || math.Abs(w.Used-99.5) > 1e-9 || w.ResetsAt == nil || w.ResetsAt.Day() != 22 || w.Aside {
		t.Errorf("weekly: %+v", w)
	}
	if w := ws[1]; w.Name != "Search · 1 hour" || w.Span != time.Hour || math.Abs(w.Used-10) > 1e-9 || w.Amount != 25 || w.Limit != 250 || w.Unit != "searches" || !w.Aside || w.ResetsAt == nil {
		t.Errorf("search: %+v", w)
	}

	// Idle planWeeklyLimit (100% remaining) omits nextRegenAt.
	_, ws, err = readSyntheticPlan([]byte(`{"planWeeklyLimit":{"nextRegenAt":"2026-11-22T00:00:00Z","percentRemaining":100,"nextRegenPercent":10}}`))
	if err != nil || len(ws) != 1 || ws[0].Used != 0 || ws[0].ResetsAt != nil {
		t.Fatalf("idle weekly: err=%v ws=%+v", err, ws)
	}

	// Previous-generation rollingFiveHourLimit + weeklyTokenLimit.
	_, ws, err = readSyntheticPlan([]byte(`{
		"rollingFiveHourLimit": {"remaining": 27, "max": 135, "nextTickAt": "2026-11-21T10:30:00Z", "tickPercent": 0.1},
		"weeklyTokenLimit": {"nextRegenAt": "2026-11-25T00:00:00Z", "percentRemaining": 75, "maxCredits": "$100.00", "remainingCredits": "$75.00"}
	}`))
	if err != nil || len(ws) != 2 {
		t.Fatalf("rolling+weekly: err=%v ws=%+v", err, ws)
	}
	if w := ws[0]; w.Name != "5 hours" || w.Span != 5*time.Hour || math.Abs(w.Used-80) > 1e-9 || w.ResetsAt == nil {
		t.Errorf("5h rolling: %+v", w)
	}
	if w := ws[1]; w.Name != "7 days" || w.Span != 7*24*time.Hour || math.Abs(w.Used-25) > 1e-9 || w.ResetsAt == nil {
		t.Errorf("weeklyTokenLimit: %+v", w)
	}

	// rollingFiveHourLimit marked limited: true reads as 100% used.
	_, ws, err = readSyntheticPlan([]byte(`{"rollingFiveHourLimit":{"remaining":5,"max":135,"nextTickAt":"2026-11-21T10:30:00Z","limited":true}}`))
	if err != nil || len(ws) != 1 || ws[0].Used != 100 {
		t.Fatalf("limited rolling: err=%v ws=%+v", err, ws)
	}

	// Documented subscription response (dev.synthetic.new/docs/synthetic/quotas).
	_, ws, err = readSyntheticPlan([]byte(`{"subscription":{"limit":135,"requests":27,"renewsAt":"2025-09-21T14:36:14.288Z"}}`))
	if err != nil || len(ws) != 1 {
		t.Fatalf("subscription: err=%v ws=%+v", err, ws)
	}
	if w := ws[0]; w.Name != "5 hours" || w.Span != 5*time.Hour || math.Abs(w.Used-20) > 1e-9 || w.Amount != 27 || w.Limit != 135 || w.Unit != "requests" || w.ResetsAt == nil || w.ResetsAt.Year() != 2025 {
		t.Errorf("subscription window: %+v", w)
	}

	// Usage-based-only key: no active subscription limit returns no windows (no card).
	for _, empty := range []string{`{}`, `{"subscription":{"limit":0,"requests":0}}`} {
		if _, ws, err := readSyntheticPlan([]byte(empty)); err != nil || len(ws) != 0 {
			t.Errorf("empty %s: err=%v ws=%+v", empty, err, ws)
		}
	}

	// Invalid JSON, null, or error payload must return an error.
	for _, bad := range []string{`null`, `not-json`, `{"error":"Invalid API Key."}`} {
		if _, _, err := readSyntheticPlan([]byte(bad)); err == nil {
			t.Errorf("bad payload %s should error", bad)
		}
	}
}

func TestSyntheticPlanQuotas(t *testing.T) {
	isolate(t)
	h := t.TempDir()
	t.Setenv("HOME", h)
	t.Setenv("USERPROFILE", h)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(h, ".config"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(h, ".cache"))
	t.Setenv("PATH", h)
	for _, v := range agentenv.Vars {
		t.Setenv(v, "")
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get("X-Host") + r.URL.Path + " " + r.Header.Get("Authorization") {
		case "api.synthetic.new/v2/quotas Bearer syn-sub":
			w.Write([]byte(`{"planWeeklyLimit":{"nextRegenAt":"2026-11-28T00:00:00Z","percentRemaining":60,"nextRegenPercent":10}}`))
		case "api.synthetic.new/v2/quotas Bearer syn-payg":
			w.Write([]byte(`{}`))
		default:
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":"Invalid API Key."}`))
		}
	}))
	defer srv.Close()
	old := http.DefaultClient.Transport
	http.DefaultClient.Transport = rewrite{srv}
	t.Cleanup(func() { http.DefaultClient.Transport = old })

	sub, err := FromPreset("synthetic")
	if err != nil {
		t.Fatal(err)
	}
	sub.Key = "syn-sub"
	if err := Save(sub); err != nil {
		t.Fatal(err)
	}
	payg := sub
	payg.ID = "synthetic-payg"
	payg.Key = "syn-payg"
	if err := Save(payg); err != nil {
		t.Fatal(err)
	}

	got := map[string]SubscriptionQuota{}
	for _, q := range PlanQuotas(context.Background()) {
		got[q.Provider] = q
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 Synthetic plan card, got %+v", got)
	}
	q := got["synthetic"]
	if q.Icon != "synthetic" || len(q.Windows) != 1 || q.Windows[0].Name != "7 days" || math.Abs(q.Windows[0].Used-40) > 1e-9 {
		t.Fatalf("synthetic quota card: %+v", q)
	}
}
