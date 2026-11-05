// Algorithm flowchart of the vortex step method with a free wake (paper 1, Fig. 2). Editable PowerPoint figure, single column.
// Owner request 2026-10-08 16:00: less text, Times New Roman.
// English v7 variant (2026-10-09, compressed Renewable Energy manuscript). Equation numbers of Section 2 of v7:
// (1)-(3) panels, (4) induced velocity, (5) core, (6) velocity split, (7) 2D term, (8) closure, (9) Jacobian,
// (10) descent condition, (11) acceptance, (12) loads, (13) DAE, (14) Euler convection, (15) shedding.
// The Chinese v5 keeps make_flowchart.js / fig03_flowchart.png with its own numbering.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
const W = 3.37, H = 3.97;
pres.defineLayout({ name: "FIG", width: W, height: H });
pres.layout = "FIG";
pres.author = "Yinong Tian";
pres.title = "OpenFAWE VSM-F algorithm";
pres.theme = { headFontFace: "Times New Roman", bodyFontFace: "Times New Roman" };

const FONT = "Times New Roman";
const INK = "1A1A1A", LINE = "4A4F57", ACC = "1F5FA8", RED = "C0392B";
const FILL = "EEF2F7", FAILFILL = "FBEDEC";
const FS = 9;
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

const X0 = 0.42, BW = 1.95, XC = X0 + BW / 2;     // main column
const XR = 2.6, RW = 0.72;                          // right column

function runs(parts) {
  return parts.map((p) => {
    if (typeof p === "string") return { text: p, options: {} };
    const [text, o] = p;
    return { text, options: { subscript: !!o.sub, superscript: !!o.sup, italic: !!o.it, bold: !!o.b } };
  });
}
function box(name, x, y, w, h, parts, opt = {}) {
  slide.addText(runs(parts), {
    objectName: name, isTextBox: true,
    shape: opt.round ? pres.shapes.ROUNDED_RECTANGLE : pres.shapes.RECTANGLE,
    rectRadius: opt.round ? 0.06 : undefined,
    x, y, w, h, fontFace: FONT, fontSize: FS, color: INK, align: "center", valign: "middle",
    fill: { color: opt.fill || FILL }, line: { color: opt.line || LINE, width: 0.75 }, margin: [1, 3, 1, 3],
  });
}
function diamond(name, x, y, w, h, parts) {
  slide.addText(runs(parts), {
    objectName: name, isTextBox: true, shape: pres.shapes.DIAMOND, x, y, w, h,
    fontFace: FONT, fontSize: FS, color: INK, align: "center", valign: "middle",
    fill: { color: "FFFFFF" }, line: { color: ACC, width: 0.9 }, margin: 0,
  });
}
function arrow(name, x1, y1, x2, y2, color = LINE, head = true) {
  slide.addShape(pres.shapes.LINE, {
    objectName: name, x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 0.9, endArrowType: head ? "triangle" : undefined },
  });
}
function label(name, x, y, w, h, parts, opt = {}) {
  slide.addText(runs(parts), { objectName: name, isTextBox: true, x, y, w, h, fontFace: FONT, fontSize: FS,
    color: opt.color || INK, align: opt.align || "left", valign: "middle", margin: 0, rotate: opt.rotate || 0 });
}
const it = (t) => [t, { it: true }];
const sub = (t) => [t, { sub: true }];

// input and start
box("Input", 0.12, 0.04, W - 0.24, 0.27, [
  "Geometry, polars, ", it("U"), sub("∞"), ", Δ", it("t"), ", ", it("N"), ", ", it("r"), sub("c"),
], { round: true, fill: "FFFFFF", line: ACC });
arrow("a_in", XC, 0.31, XC, 0.44);
box("Start", X0, 0.44, BW, 0.27, [it("t"), " = 0: bound circulation, no wake"], { round: true });
arrow("a_start", XC, 0.71, XC, 0.84);
// circulation
const yInd = 0.84, hInd = 0.27;
box("Induction", X0, yInd, BW, hInd, ["¾‑chord velocity, Eqs. (6), (7)"]);
arrow("a_ind", XC, yInd + hInd, XC, 1.24);
box("Newton", X0, 1.24, BW, 0.27, ["Newton solve for ", ["Γ", { b: true }], ", Eqs. (8)–(10)"]);
arrow("a_newton", XC, 1.51, XC, 1.64);
const dW = 1.5, dH = 0.46;
diamond("Converged", XC - dW / 2, 1.64, dW, dH, ["Eq. (11) met?"]);
arrow("a_fail", XC + dW / 2, 1.87, XR, 1.87, RED);
label("no1", XC + dW / 2 + 0.03, 1.70, 0.3, 0.15, ["no"], { color: RED });
box("Failed", XR, 1.73, RW, 0.28, ["Run fails"], { fill: FAILFILL, line: RED });
arrow("a_yes1", XC, 2.10, XC, 2.24);
label("yes1", XC + 0.09, 2.08, 0.3, 0.15, ["yes"]);
box("Loads", X0, 2.24, BW, 0.27, ["Loads, Eq. (12)"]);
arrow("a_loads", XC, 2.51, XC, 2.64);
diamond("LastStep", XC - dW / 2, 2.64, dW, dH, [it("n"), " = ", it("N"), "?"]);
arrow("a_out", XC + dW / 2, 2.87, XR, 2.87, ACC);
label("yes2", XC + dW / 2 + 0.03, 2.70, 0.3, 0.15, ["yes"], { color: ACC });
box("Output", XR, 2.66, RW, 0.42, ["Loads and wake states"], { round: true, fill: "FFFFFF", line: ACC });
arrow("a_no2", XC, 3.10, XC, 3.24);
label("no2", XC + 0.09, 3.08, 0.3, 0.15, ["no"]);
// wake
box("Convection", X0, 3.24, BW, 0.27, ["Convect wake, Eqs. (13), (14)"]);
arrow("a_conv", XC, 3.51, XC, 3.64);
box("Shedding", X0, 3.64, BW, 0.27, ["Shed a wake lattice row, Eq. (15)"]);
// loop back
const xl = 0.22;
arrow("loop1", X0, 3.775, xl, 3.775, ACC, false);
arrow("loop2", xl, 3.775, xl, yInd + hInd / 2, ACC, false);
arrow("loop3", xl, yInd + hInd / 2, X0, yInd + hInd / 2, ACC, true);
label("looplabel", 0.11 - 0.45, 2.3 - 0.08, 0.9, 0.16, [it("n"), " ← ", it("n"), " + 1"], { color: ACC, align: "center", rotate: 270 });

pres.writeFile({ fileName: "fig03_flowchart_en7.pptx" }).then(() => console.log("written"));
