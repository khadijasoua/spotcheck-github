"""SpotCheck demo website (Gradio).

Shows, for a chosen test image, the referral recommendation, the Grad-CAM heatmap,
the melanoma probability against the referral threshold and all class probabilities.
A live threshold slider shows sensitivity versus false referrals on the whole test set.

Usage: python app.py   then open http://127.0.0.1:7860   (Ctrl + C to stop)
Start-up takes about a minute: all 2,005 test images are scored once.
"""

# ---------------------------------------------------------------
# 1. TOOLS
# ---------------------------------------------------------------
import base64
import inspect
import io
import random

import numpy as np
import gradio as gr                                   # builds the website
from PIL import Image                                 # to resize pictures
from heatmap import explain, test_raw, model          # our heatmap code, test photos, trained model
from common import get_data, get_probs, MEL


# ---------------------------------------------------------------
# 2. SETTINGS: change names and contact details here
# ---------------------------------------------------------------
THRESHOLD = 0.2                       # the alarm level chosen in threshold.py
TEAM = ["Khadija Soua", "Robin Harris Subiakno"]
COURSE = "Introduction to AI · Winter Semester 2026"
SUPPORT_EMAIL = "support@spotcheck.example"    # demo address, replace with a real one
SUPPORT_PHONE = "+49 000 0000000"               # demo number, replace with a real one

# Colours: white page, navy accent, red/green only for the verdict
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
NAVY, BLUE = "#1e3a8a", "#2563eb"
INK, BODY, MUTED, LINE, SOFT = "#0f172a", "#334155", "#64748b", "#e2e8f0", "#f8fafc"
RED, GREEN, AMBER, GREY = "#dc2626", "#16a34a", "#d97706", "#cbd5e1"


# ---------------------------------------------------------------
# 3. CLINICAL INFO FOR THE 7 TYPES
# Same order as CLASS_NAMES: (name, is it malignant/premalignant?, short note for a GP)
# ---------------------------------------------------------------
TYPES = [
    ("Actinic keratosis / IEC", True, "Sun-damage lesion; intraepithelial carcinoma is an early skin cancer."),
    ("Basal cell carcinoma", True, "Most common skin cancer. Slow-growing, rarely metastasises; needs treatment."),
    ("Benign keratosis", False, "Seborrhoeic keratosis, solar lentigo or LPLK. Can mimic melanoma."),
    ("Dermatofibroma", False, "Benign fibrous nodule, typically on the lower legs."),
    ("Melanoma", True, "Malignant melanocytic tumour. Early excision is decisive for prognosis."),
    ("Melanocytic naevus", False, "Common benign mole. Most frequent type in the data (67%)."),
    ("Vascular lesion", False, "Benign vascular lesion, e.g. cherry angioma. Usually red or purple."),
]

# Demo cases chosen in heatmap.py: button text -> test photo number
DEMOS = {
    "Benign naevus": 717,
    "Clear melanoma": 15,
    "Borderline case": 12,
    "Missed melanoma": 158,
}


# ---------------------------------------------------------------
# 4. ANALYSE ALL TEST PHOTOS ONCE (for the live alarm-level slider)
# ---------------------------------------------------------------
print("Analysing all 2,005 test photos once. This takes about a minute...")
_probs, _labels = get_probs(model, get_data("test"))
P_MEL = _probs[:, MEL].numpy()                 # melanoma probability of every test photo
IS_MEL = _labels.numpy() == MEL                # True = really melanoma
N_MEL, N_OTHER = int(IS_MEL.sum()), int((~IS_MEL).sum())
print(f"Done: {N_MEL} melanomas and {N_OTHER} other photos.")


def stats(t):
    """For alarm level t: melanomas caught, missed, and healthy spots flagged."""
    alarm = P_MEL >= t
    caught = int((alarm & IS_MEL).sum())
    false_alarms = int((alarm & ~IS_MEL).sum())
    return caught, N_MEL - caught, false_alarms


CURVE = [stats(t) for t in np.round(np.arange(0.01, 0.96, 0.01), 2)]   # the whole trade-off curve


# ---------------------------------------------------------------
# 5. SMALL HTML HELPERS
# ---------------------------------------------------------------
def to_b64(img):
    """Turn a picture into text that can sit directly inside the web page."""
    buf = io.BytesIO()
    img.resize((360, 360)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def card(inner, pad="20px 22px", extra=""):
    return (f'<div style="background:#fff;border:1px solid {LINE};border-radius:16px;padding:{pad};'
            f'box-shadow:0 1px 3px rgba(15,23,42,.05);{extra}">{inner}</div>')


def label(text, colour=MUTED):
    return (f'<div style="font-size:12px;font-weight:600;letter-spacing:.07em;color:{colour};'
            f'margin-bottom:10px">{text}</div>')


# ---------------------------------------------------------------
# 6. THE RESULT FOR ONE PHOTO
# ---------------------------------------------------------------
def report_html(case, t):
    p = case["probs"]
    p_mel = p[MEL]
    refer = p_mel >= t
    name, malignant, note_text = TYPES[case["pred"]]

    # a) The verdict
    if refer:
        bg, edge, ink, icon = "#fef2f2", RED, "#7f1d1d", "&#9888;"
        title, sub = "Refer to dermatology", "Melanoma probability is at or above the alarm level."
    else:
        bg, edge, ink, icon = "#f0fdf4", GREEN, "#14532d", "&#10003;"
        title, sub = "No referral indicated", ("Below the alarm level. Routine monitoring; "
                                               "re-assess if size, shape or colour change.")
    verdict = (f'<div style="background:{bg};border:1px solid {edge};border-left:6px solid {edge};'
               f'border-radius:16px;padding:22px 24px">'
               f'<div style="font-size:12px;font-weight:600;letter-spacing:.07em;color:{ink};opacity:.75">'
               f'RECOMMENDATION</div>'
               f'<div style="font-size:28px;font-weight:700;color:{ink};margin:4px 0 4px">{icon}&nbsp; {title}</div>'
               f'<div style="font-size:15px;color:{ink};line-height:1.5">{sub}</div></div>')

    # b) Melanoma probability bar with the alarm line
    gauge = card(
        f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
        f'{label("MELANOMA PROBABILITY")}'
        f'<div style="font-size:24px;font-weight:700;color:{edge}">{p_mel:.0%}</div></div>'
        f'<div style="position:relative;height:12px;background:#f1f5f9;border-radius:6px">'
        f'<div style="width:{max(p_mel * 100, 1):.1f}%;height:12px;background:{edge};border-radius:6px"></div>'
        f'<div style="position:absolute;left:{t * 100:.1f}%;top:-6px;height:24px;width:2px;background:{NAVY}"></div>'
        f'</div>'
        f'<div style="position:relative;height:22px;font-size:12px;color:{MUTED}">'
        f'<span style="position:absolute;left:{t * 100:.1f}%;top:8px;transform:translateX(-50%);'
        f'white-space:nowrap;color:{NAVY};font-weight:600">alarm level {t:.0%}</span></div>',
        extra="margin-top:14px")

    # c) Most likely type
    tag_bg, tag_ink, tag = (("#fee2e2", "#991b1b", "Malignant / premalignant") if malignant
                            else ("#f1f5f9", BODY, "Benign"))
    likely = card(
        label("MOST LIKELY DIAGNOSIS")
        + f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px">'
          f'<div style="font-size:21px;font-weight:700;color:{INK}">{name}</div>'
          f'<div style="font-size:15px;font-weight:600;color:{BODY}">{p[case["pred"]]:.0%}</div></div>'
          f'<span style="display:inline-block;margin-top:8px;font-size:12px;font-weight:600;padding:3px 10px;'
          f'border-radius:999px;background:{tag_bg};color:{tag_ink}">{tag}</span>'
          f'<div style="font-size:14px;color:{BODY};line-height:1.55;margin-top:8px">{note_text}</div>',
        extra="margin-top:14px")

    # d) Extra warning: no melanoma alarm, but the top answer is another malignant type
    warn = ""
    if not refer and malignant:
        warn = (f'<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:14px;padding:12px 16px;'
                f'margin-top:14px;color:#78350f;font-size:14px;line-height:1.5">'
                f'<b style="color:#78350f">Consider referral:</b> no melanoma alarm, but the most likely '
                f'diagnosis ({name}) is malignant or premalignant.</div>')

    # e) All 7 types as bars
    rows = ""
    for i in sorted(range(7), key=lambda k: -p[k]):
        n, m, _ = TYPES[i]
        rows += (f'<div style="display:flex;align-items:center;gap:12px;margin:7px 0;font-size:14px">'
                 f'<div style="width:170px;color:{INK}">{n}</div>'
                 f'<div style="flex:1;height:8px;background:#f1f5f9;border-radius:4px">'
                 f'<div style="width:{p[i] * 100:.1f}%;height:8px;border-radius:4px;'
                 f'background:{RED if m else GREY}"></div></div>'
                 f'<div style="width:40px;text-align:right;color:{BODY}">{p[i]:.0%}</div></div>')
    bars = card(label("ALL 7 TYPES") + rows
                + f'<div style="font-size:12px;color:{MUTED};margin-top:10px">'
                  f'<span style="color:{RED}">&#9679;</span> malignant / premalignant &nbsp;&nbsp;'
                  f'<span style="color:{GREY}">&#9679;</span> benign</div>',
                extra="margin-top:14px")

    # f) Photo and heatmap, side by side
    def figure(b64, title, sub):
        return (f'<div><img src="data:image/png;base64,{b64}" style="width:100%;border-radius:12px;display:block">'
                f'<div style="font-size:14px;font-weight:600;color:{INK};margin-top:10px">{title}</div>'
                f'<div style="font-size:13px;color:{MUTED}">{sub}</div></div>')
    pictures = card(
        label(f"TEST PHOTO #{case['index']}")
        + f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">'
          f'{figure(case["img"], "Dermatoscope image", "As seen by the GP")}'
          f'{figure(case["heat"], "Where the AI looked", "Red = most influence on the result")}</div>'
          f'<div style="background:{SOFT};border-radius:10px;padding:10px 14px;margin-top:14px;'
          f'font-size:13px;color:{MUTED};line-height:1.5">'
          f'For evaluation: the dataset label is <b style="color:{BODY}">{TYPES[case["true"]][0]}</b>. '
          f'{"&#10003; Top prediction matches." if case["true"] == case["pred"] else "&#10007; Top prediction differs."}'
          f'</div>')

    left = f'<div>{pictures}</div>'
    right = f'<div>{verdict}{gauge}{likely}{warn}{bars}</div>'
    return (f'<div style="font-family:{FONT};display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));'
            f'gap:16px;align-items:start">{left}{right}</div>')


# ---------------------------------------------------------------
# 7. THE ALARM-LEVEL PANEL (all test photos)
# ---------------------------------------------------------------
def tradeoff_html(t):
    caught, missed, fa = stats(t)
    alarms = caught + fa

    def tile(value, text, sub, colour):
        return card(f'<div style="font-size:30px;font-weight:700;color:{colour};line-height:1.1">{value}</div>'
                    f'<div style="font-size:14px;font-weight:600;color:{INK};margin-top:6px">{text}</div>'
                    f'<div style="font-size:13px;color:{MUTED}">{sub}</div>', pad="16px 18px")

    tiles = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px">'
             + tile(f"{caught / N_MEL:.0%}", "Melanomas detected", f"{caught} of {N_MEL}", GREEN)
             + tile(str(missed), "Melanomas missed", "False negatives", RED)
             + tile(f"{fa / N_OTHER:.0%}", "Benign lesions referred", f"{fa} of {N_OTHER}", AMBER)
             + tile(f"1 in {alarms / caught:.1f}" if caught else "–", "Referrals that are melanoma",
                    "Positive predictive value", NAVY)
             + '</div>')

    # The chart: x = benign lesions referred (%), y = melanomas detected (%)
    W, H, L, R, T_, B = 760, 300, 56, 24, 18, 48
    X = lambda v: L + v / 60 * (W - L - R)
    Y = lambda v: H - B - v / 100 * (H - T_ - B)
    pts = [(X(min(f / N_OTHER * 100, 60)), Y(c / N_MEL * 100)) for c, _, f in CURVE]
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    grid = ""
    for v in range(0, 101, 25):
        grid += (f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#f1f5f9"/>'
                 f'<text x="{L - 10}" y="{Y(v) + 4:.1f}" font-size="12" fill="{MUTED}" text-anchor="end">{v}%</text>')
    for v in range(0, 61, 10):
        grid += f'<text x="{X(v):.1f}" y="{H - B + 20}" font-size="12" fill="{MUTED}" text-anchor="middle">{v}%</text>'
    c0, _, f0 = stats(THRESHOLD)
    ox, oy = X(min(f0 / N_OTHER * 100, 60)), Y(c0 / N_MEL * 100)
    cx, cy = X(min(fa / N_OTHER * 100, 60)), Y(caught / N_MEL * 100)
    left_side = cx > 260
    svg = (f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;font-family:{FONT}">{grid}'
           f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="{GREY}"/>'
           f'<path d="{path}" fill="none" stroke="{NAVY}" stroke-width="2.5" stroke-linejoin="round"/>'
           f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="6" fill="#fff" stroke="{MUTED}" stroke-width="2"/>'
           f'<text x="{ox + 10:.1f}" y="{oy + 20:.1f}" font-size="12" fill="{MUTED}">recommended: {THRESHOLD:.0%}</text>'
           f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{BLUE}" stroke="#fff" stroke-width="3"/>'
           f'<text x="{cx - 12 if left_side else cx + 12:.1f}" y="{cy - 12:.1f}" font-size="13" font-weight="600" '
           f'fill="{BLUE}" text-anchor="{"end" if left_side else "start"}">selected: {t:.0%}</text>'
           f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" font-size="13" fill="{BODY}" text-anchor="middle">'
           f'benign lesions referred</text>'
           f'<text x="14" y="{(T_ + H - B) / 2:.0f}" font-size="13" fill="{BODY}" text-anchor="middle" '
           f'transform="rotate(-90 14 {(T_ + H - B) / 2:.0f})">melanomas detected</text></svg>')

    return (f'<div style="font-family:{FONT}">{tiles}'
            f'<div style="margin-top:14px">{card(svg, pad="14px 16px 6px")}</div></div>')


# ---------------------------------------------------------------
# 8. WHAT HAPPENS ON EACH CLICK
# ---------------------------------------------------------------
def load_case(index):
    """Predict one photo and make its heatmap (the slow part, once per photo)."""
    img, overlay, true, pred, probs = explain(int(index))
    return {"index": int(index), "img": to_b64(img), "heat": to_b64(Image.fromarray(overlay)),
            "true": true, "pred": pred, "probs": [float(v) for v in probs]}


def pick_demo(choice):
    return load_case(DEMOS.get(choice, 717))


def pick_random():
    return load_case(random.randint(0, len(test_raw) - 1))


def render(case, t):
    """Draw the result and the alarm panel (fast, runs on every slider move)."""
    return (report_html(case, t) if case else ""), tradeoff_html(t)


# ---------------------------------------------------------------
# 9. THE PAGE
# ---------------------------------------------------------------
NAV = f"""
<div style="font-family:{FONT};display:flex;justify-content:space-between;align-items:center;
            padding:14px 4px;border-bottom:1px solid {LINE};flex-wrap:wrap;gap:10px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:34px;height:34px;border-radius:10px;background:{NAVY};display:flex;
                align-items:center;justify-content:center">
      <div style="width:14px;height:14px;border-radius:50%;border:3px solid #fff"></div></div>
    <div style="font-size:20px;font-weight:700;color:{INK};letter-spacing:-.01em">SpotCheck</div>
  </div>
  <div style="display:flex;align-items:center;gap:14px;font-size:13px;color:{MUTED}">
    <span style="color:{MUTED}">Skin-lesion triage support for GPs</span>
    <span style="background:#eff6ff;color:{NAVY};padding:5px 12px;border-radius:999px;font-weight:600">
      Decision support · not a medical device</span>
  </div>
</div>"""


def section(title, sub):
    return (f'<div style="font-family:{FONT};margin:10px 2px 2px">'
            f'<div style="font-size:22px;font-weight:700;color:{INK}">{title}</div>'
            f'<div style="font-size:14px;color:{MUTED};margin-top:4px">{sub}</div></div>')


HOW = f"""
<div style="font-family:{FONT};display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px">
  {''.join(card(f'<div style="width:30px;height:30px;border-radius:9px;background:#eff6ff;color:{NAVY};'
                f'font-weight:700;display:flex;align-items:center;justify-content:center">{n}</div>'
                f'<div style="font-weight:600;color:{INK};margin:12px 0 4px">{t}</div>'
                f'<div style="font-size:14px;color:{BODY};line-height:1.55">{d}</div>')
           for n, t, d in [
               ("1", "Image", "A dermatoscope image of a single lesion."),
               ("2", "Classification", "A ResNet-18 network estimates the probability of 7 lesion types."),
               ("3", "Alarm level", f"Referral is suggested when the melanoma probability reaches "
                                    f"{THRESHOLD:.0%}, a low level on purpose: a missed melanoma costs more "
                                    f"than an extra referral."),
               ("4", "Clinical judgement", "The heatmap and probabilities support the GP's decision; "
                                           "they never replace it."),
           ])}
</div>"""

FOOTER = f"""
<div style="font-family:{FONT};border-top:1px solid {LINE};margin-top:8px;padding:26px 4px 6px;
            display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:28px;font-size:14px;line-height:1.65">
  <div>
    <div style="font-weight:700;color:{INK};font-size:16px">SpotCheck</div>
    <div style="color:{MUTED};margin-top:6px">A student project on AI-supported skin-cancer triage.
      Not a medical device; not for diagnosing real patients. Detects about 4 in 5 melanomas in our tests,
      was trained mostly on light skin, and works only on dermatoscope images.</div>
  </div>
  <div>
    <div style="font-weight:700;color:{INK}">Team</div>
    {''.join(f'<div style="color:{BODY}">{n}</div>' for n in TEAM)}
    <div style="color:{MUTED};margin-top:8px">{COURSE}</div>
  </div>
  <div>
    <div style="font-weight:700;color:{INK}">Support</div>
    <div style="color:{BODY}">Email: <a href="mailto:{SUPPORT_EMAIL}" style="color:{BLUE}">{SUPPORT_EMAIL}</a></div>
    <div style="color:{BODY}">Phone: {SUPPORT_PHONE}</div>
    <div style="color:{MUTED};margin-top:8px">Data: DermaMNIST (MedMNIST v2), based on HAM10000.
      Images are processed locally and never stored.</div>
  </div>
</div>"""

# Restyle Gradio's own buttons, pills and sliders so they match the page.
THEME = gr.themes.Base(
    primary_hue="blue", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
_look = dict(
    body_background_fill="#ffffff",
    block_background_fill="#ffffff", block_border_color=LINE, block_border_width="1px",
    block_shadow="0 1px 3px rgba(15,23,42,.05)", block_radius="16px",
    block_label_background_fill="#ffffff", block_label_text_color=MUTED, block_title_text_color=INK,
    block_label_border_width="0px",
    button_primary_background_fill=NAVY, button_primary_background_fill_hover="#1e40af",
    button_primary_text_color="#ffffff", button_primary_border_color=NAVY,
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_hover="#eff6ff",
    button_secondary_text_color=NAVY, button_secondary_border_color="#bfdbfe",
    slider_color=NAVY,
    checkbox_label_background_fill="#ffffff", checkbox_label_background_fill_hover="#eff6ff",
    checkbox_label_background_fill_selected=NAVY, checkbox_label_text_color=INK,
    checkbox_label_text_color_selected="#ffffff", checkbox_label_border_color="#cbd5e1",
    checkbox_label_border_color_selected=NAVY, checkbox_label_border_width="1px",
    input_background_fill="#ffffff", input_border_color=LINE,
)
THEME = THEME.set(**{k: v for k, v in _look.items() if hasattr(THEME, k)})   # skip names your version lacks

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto !important; background: #fff !important;}
footer {display: none !important;}
.pills input[type=radio] {display: none !important;}                       /* pills without radio dots */
.pills label {border-radius: 999px !important; padding: 8px 16px !important; cursor: pointer;}
.controls {border: 1px solid #e2e8f0 !important; border-radius: 16px !important; padding: 12px 16px !important;
           background: #f8fafc !important; align-items: center !important;}
.controls .block, .controls .form {background: transparent !important; border: none !important; box-shadow: none !important;}
"""

# Always show the page in light mode (dark mode made the text unreadable).
FORCE_LIGHT = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.href = url.href;
  }
}
"""

# Newer Gradio versions moved theme, css and js from Blocks() to launch().
# This checks which one your version has, so the code works with both.
_style = {"theme": THEME, "css": CSS, "js": FORCE_LIGHT}
_blocks_params = inspect.signature(gr.Blocks.__init__).parameters
_launch_params = inspect.signature(gr.Blocks.launch).parameters
blocks_options = {k: v for k, v in _style.items() if k in _blocks_params}
launch_options = {k: v for k, v in _style.items() if k not in blocks_options and k in _launch_params}

with gr.Blocks(title="SpotCheck", **blocks_options) as demo:
    case = gr.State()        # the photo currently shown (so the slider doesn't redo the heatmap)
    gr.HTML(NAV)

    # ---------- Small control bar above the result ----------
    with gr.Row(elem_classes="controls", equal_height=True):
        picker = gr.Radio(list(DEMOS), value="Benign naevus", show_label=False,
                          elem_classes="pills", scale=5)
        random_button = gr.Button("Random case", variant="secondary", size="sm", scale=1, min_width=130)
    with gr.Accordion("Choose any test photo by number", open=False):
        with gr.Row(equal_height=True):
            photo_slider = gr.Slider(0, len(test_raw) - 1, step=1, value=717, label="Test photo", scale=4)
            analyse_button = gr.Button("Analyse", variant="primary", scale=1, min_width=120)

    # ---------- The result, first and biggest ----------
    report = gr.HTML()

    # ---------- The alarm level ----------
    gr.HTML(section("Alarm level", "Drag to change when SpotCheck suggests a referral. The recommendation "
                                   "above and the results for all 2,005 test images update live."))
    alarm_slider = gr.Slider(0.05, 0.60, step=0.01, value=THRESHOLD, label="Refer when melanoma probability is at least")
    tradeoff = gr.HTML()

    # ---------- How it works + footer ----------
    gr.HTML(section("How SpotCheck works", "From image to referral decision in four steps."))
    gr.HTML(HOW)
    gr.HTML(FOOTER)

    # ---------- What the controls do ----------
    views = [report, tradeoff]
    picker.change(pick_demo, picker, case).then(render, [case, alarm_slider], views)
    random_button.click(pick_random, None, case).then(render, [case, alarm_slider], views)
    analyse_button.click(load_case, photo_slider, case).then(render, [case, alarm_slider], views)
    alarm_slider.change(render, [case, alarm_slider], views)       # live: no new heatmap needed
    demo.load(pick_demo, picker, case).then(render, [case, alarm_slider], views)

demo.launch(**launch_options)   # start the website
