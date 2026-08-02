# 08 — Dashboard UI/UX Design (Streamlit)

Design direction developed with the frontend-design skill's process (brainstorm tokens → critique against generic defaults → commit). Note: the brief referenced a "UI-UX-Pro-Max" skill, which is not installed in this environment; the installed `frontend-design` skill was used and its guidance is baked in below.

## 1. Design concept — "Mission Control for Money"

The subject is *cost telemetry for AI infrastructure*: the aesthetic vocabulary comes from trading terminals and flight-deck instrumentation, not generic SaaS. Dense, confident numerals; instrument-panel cards; one signature moment.

**Signature element:** the **Savings Meter** — a large animated odometer-style counter on the analytics home showing cumulative dollars saved, ticking upward as new data arrives, with a thin dual-line "actual vs. counterfactual" spark underneath. Everything else stays quiet so this one element carries the page.

### Design tokens

| Token | Value | Use |
|---|---|---|
| `--bg` | `#0B0E14` | app background (deep blue-black, not pure black) |
| `--surface` | `rgba(22, 27, 38, 0.66)` | glass cards (blur 14px, 1px `rgba(255,255,255,.08)` border) |
| `--ink` | `#E7EBF0` | primary text |
| `--ink-dim` | `#8A93A6` | secondary text |
| `--save` | `#4ADE9C` | savings, pass, money-positive (mint, not acid-green) |
| `--spend` | `#F2B441` | baseline cost, warnings (amber) |
| `--fail` | `#F26D6D` | failures, escalations |
| `--accent` | `#6C8CFF` | interactive, links, selected states (periwinkle) |

Typography: **Space Grotesk** for display/headlines (technical character without novelty), **Inter** for body/UI, **JetBrains Mono** for all numerals, costs, ids, and log rows — tabular figures make the money data feel like an instrument readout. Type scale 12/14/16/20/28/44; the Savings Meter alone gets 64px.

Layout: 12-col fluid grid, 24px gutters; cards on an 8px spacing system; sidebar navigation (Streamlit native, restyled) with icon + label. Dark mode is the *primary* theme (light optional later). Motion: 180 ms ease-out on card hover (lift + border brighten), number count-up on load (respecting `prefers-reduced-motion`), skeleton shimmer for loading. No scattered animation — the meter is the one orchestrated moment.

**Anti-generic check (per skill):** avoided the cream+serif+terracotta default, avoided pure-black + acid-green terminal cliché, avoided newspaper hairlines. Palette and type were chosen for this subject (financial telemetry); the risk taken is the odometer signature + mono-numeral discipline.

## 2. Streamlit implementation strategy

Streamlit can look premium if you treat it as a layout engine and own the CSS: one `theme.css` injected once (tokens above as CSS custom properties, `color-scheme: dark`), `st.markdown` component templates for cards/badges/stats (HTML strings from `components/`), Plotly charts with a shared custom template (transparent paper, token colors, Inter labels, unified hover), `st.container`-based grids, and `.streamlit/config.toml` base theme matched to tokens so native widgets don't clash. Custom scrollbar, hidden default header/footer, wide layout.

## 3. Pages

### 3.1 Overview (landing / analytics home)
- Hero row: **Savings Meter** (signature) + three glass stat cards — Net savings % (after verification cost), Requests (7d), Avg quality score with pass-rate ring.
- Dual-line area chart: actual spend vs. counterfactual baseline, shaded gap labeled "saved". Range switcher 24h/7d/30d.
- Routing distribution donut (model share) + escalation-rate sparkline row.
- Alert banner slot at top (from `alerts` table) — amber/red glass strip.

### 3.2 Request Playground
- Left: prompt editor (mono), task-type select, temperature/max-tokens, "force tier" (admin), Send.
- Right: response pane with a **routing receipt** card: classifier tier + confidence bar, chosen model chip, cost vs. baseline ("this request cost $0.0004 — GPT-4o would have cost $0.0049"), latency, cache/escalation badges.
- Below: last 10 playground calls as compact receipt rows. Empty state: three sample-prompt buttons (one per tier) inviting a first run.

### 3.3 Cost Savings
- The money page: cumulative savings area chart; savings by model table (mono numerals, right-aligned); cache savings line item; verification spend line item; **net** savings headline. "What-if" panel: recompute savings under a different tier→model map (client-side math from pricing) — a genuinely impressive interactive touch.

### 3.4 Routing Explorer
- Sankey: requests → predicted tier → effective tier → model (shows confidence bumps and fallbacks visually).
- Confusion matrix heatmap (classifier vs. verifier-corrected labels), confidence-distribution histogram per tier, classifier version timeline with metric deltas.

### 3.5 Logs Explorer
- Filter bar (model, tier, verdict, escalated, date range) → virtualized table, mono, dense rows: time, id (copy), tier chip, model chip, tokens, cost, latency, quality dot.
- Row click → detail drawer: full lifecycle timeline (received → classified → routed → responded → verified), feature vector, judge rationale, escalation events. This is the "audit trail" demo moment.

### 3.6 Model Registry
- Card per model: provider logo mark, pricing table, quality tier, live breaker state (green/amber/red dot with pulse when open), avg latency sparkline, active toggle (admin).

### 3.7 Settings
- Routing config editor: tier cards with primary/fallback selects + validation messages inline; comment field required; "Apply" creates a new version (shows diff before confirm). Config version history list with rollback buttons. Sampling policy sliders (base rate, canary hours, daily budget). API keys management (create → raw key shown once in a copy-card).

## 4. States (every page, by design not accident)

- **Loading:** skeleton cards with shimmer matching final layout (no spinners on data areas).
- **Empty:** instructive, action-forward — e.g. Logs: "No requests yet. Send one from the Playground →" with a button. Never a blank chart.
- **Error:** glass card in `--fail` tint: what failed ("Couldn't reach the database"), and the fix ("Check that the postgres container is running"), plus retry.
- **Success:** toast bottom-right ("Routing config v8 applied — canary verification active for 24h"), consistent verb tense with the button that caused it.

## 5. Responsiveness & accessibility

Cards stack to single column < 900px; tables gain horizontal scroll with pinned first column; Plotly `responsive=True`. Contrast: all text ≥ 4.5:1 against `--bg` (dim ink checked at 4.6:1); focus rings (`--accent`, 2px) on all interactives; reduced-motion media query disables count-ups and shimmer; charts never encode meaning by color alone (patterns/labels double-encode pass/fail).
