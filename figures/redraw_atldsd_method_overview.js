const fs = require("fs");
const path = require("path");

const OUT_DIR = "D:\\Pictures";
const TARGET = path.join(OUT_DIR, "ATLDSD_final_module_architecture.drawio");
const PACK = path.join(OUT_DIR, "ATLDSD_paper_figures.drawio");
const REPO_COPY = path.join("figures", "drawio", "ATLDSD_final_module_architecture.drawio");
const REPO_PACK = path.join("figures", "drawio", "ATLDSD_paper_figures.drawio");

const agent =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/30.0.4 Chrome/144.0.7559.236 Electron/40.8.4 Safari/537.36";

const C = {
  ink: "#0F172A",
  muted: "#64748B",
  hair: "#CBD5E1",
  line: "#334155",
  bg: "#FFFFFF",
  band: "#F8FAFC",
  inputFill: "#ECFDF5",
  input: "#10B981",
  encFill: "#EFF6FF",
  enc: "#2563EB",
  contextFill: "#ECFEFF",
  context: "#0891B2",
  fusionFill: "#F5F3FF",
  fusion: "#7C3AED",
  repFill: "#FFFBEB",
  rep: "#D97706",
  lossFill: "#FFF7ED",
  loss: "#EA580C",
  outFill: "#DCFCE7",
  out: "#16A34A",
  warnFill: "#FFF1F2",
  warn: "#E11D48",
};

let n = 2000;
const id = (p) => `${p}-${++n}`;
const esc = (s) => String(s)
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;")
  .replace(/\n/g, "&#xa;");

function style(o) {
  return Object.entries(o)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${v}`)
    .join(";") + ";";
}

const cells = [];

function rect(value, x, y, w, h, fill, stroke, opt = {}) {
  const cid = id("b");
  cells.push(`<mxCell id="${cid}" value="${esc(value)}" parent="1" vertex="1" style="${style({
    rounded: 1,
    whiteSpace: "wrap",
    html: 1,
    arcSize: opt.arc || 10,
    absoluteArcSize: 1,
    fillColor: fill,
    strokeColor: stroke,
    strokeWidth: opt.strokeWidth || 1.4,
    fontFamily: "Helvetica",
    fontColor: opt.fontColor || C.ink,
    fontSize: opt.fontSize || 13,
    fontStyle: opt.bold ? 1 : undefined,
    align: opt.align || "center",
    verticalAlign: "middle",
    spacing: opt.spacing || 8,
    shadow: 0,
  })}"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry" /></mxCell>`);
  return cid;
}

function text(value, x, y, w, h, opt = {}) {
  const cid = id("t");
  cells.push(`<mxCell id="${cid}" value="${esc(value)}" parent="1" vertex="1" style="${style({
    text: "",
    strokeColor: "none",
    fillColor: "none",
    html: 1,
    fontFamily: "Helvetica",
    fontColor: opt.color || C.ink,
    fontSize: opt.size || 12,
    fontStyle: opt.bold ? 1 : undefined,
    align: opt.align || "left",
    verticalAlign: opt.valign || "middle",
    whiteSpace: "wrap",
    rounded: 0,
  })}"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry" /></mxCell>`);
  return cid;
}

function pill(value, x, y, w, h, fill, stroke, opt = {}) {
  const cid = id("p");
  cells.push(`<mxCell id="${cid}" value="${esc(value)}" parent="1" vertex="1" style="${style({
    rounded: 1,
    whiteSpace: "wrap",
    html: 1,
    arcSize: 50,
    absoluteArcSize: 1,
    fillColor: fill,
    strokeColor: stroke,
    strokeWidth: 1.2,
    fontFamily: "Helvetica",
    fontColor: opt.color || C.ink,
    fontSize: opt.size || 11,
    fontStyle: opt.bold ? 1 : undefined,
    align: "center",
    verticalAlign: "middle",
    spacing: 4,
    shadow: 0,
  })}"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry" /></mxCell>`);
  return cid;
}

function edge(source, target, opt = {}) {
  const cid = id("e");
  const label = opt.label ? ` value="${esc(opt.label)}"` : "";
  const pts = opt.points && opt.points.length
    ? `<Array as="points">${opt.points.map(([x, y]) => `<mxPoint x="${x}" y="${y}" />`).join("")}</Array>`
    : "";
  cells.push(`<mxCell id="${cid}"${label} parent="1" source="${source}" target="${target}" edge="1" style="${style({
    edgeStyle: opt.edgeStyle || "orthogonalEdgeStyle",
    rounded: 1,
    orthogonalLoop: 1,
    jettySize: "auto",
    html: 1,
    strokeColor: opt.color || C.line,
    strokeWidth: opt.width || 2,
    endArrow: opt.endArrow || "block",
    endFill: opt.endFill === undefined ? 1 : opt.endFill,
    dashed: opt.dashed ? 1 : undefined,
    dashPattern: opt.dashPattern,
    fontFamily: "Helvetica",
    fontSize: 10,
    fontColor: opt.color || C.line,
    labelBackgroundColor: "#FFFFFF",
    exitX: opt.exitX,
    exitY: opt.exitY,
    entryX: opt.entryX,
    entryY: opt.entryY,
    exitDx: 0,
    exitDy: 0,
    entryDx: 0,
    entryDy: 0,
  })}"><mxGeometry relative="1" as="geometry">${pts}</mxGeometry></mxCell>`);
  return cid;
}

function badge(num, x, y, color) {
  return pill(num, x, y, 28, 28, "#FFFFFF", color, { bold: true, size: 12, color });
}

// Header
text("Lesion-focused DeepLabV3+ for ATLDSD", 70, 34, 690, 34, { size: 26, bold: true });
text("Single-stage six-class semantic segmentation with context refinement, cross-scale fusion, deployable decoder and lesion-only Dice supervision", 72, 72, 980, 24, { size: 12, color: C.muted });
pill("dual-seed avg\nmIoU 77.10", 1095, 35, 130, 46, C.outFill, C.out, { bold: true });
pill("FG mIoU\n72.83", 1240, 35, 105, 46, "#E0F2FE", "#0284C7", { bold: true });
pill("lesion avg IoU\n67.14", 1360, 35, 120, 46, C.lossFill, C.loss, { bold: true });

// Main containers
rect("", 60, 130, 1420, 430, C.band, C.hair, { strokeWidth: 1, arc: 14 });
text("A  Overall architecture", 86, 150, 230, 24, { size: 15, bold: true });
text("Encoder", 294, 178, 150, 20, { size: 11, bold: true, color: C.enc, align: "center" });
text("Context head", 650, 178, 180, 20, { size: 11, bold: true, color: C.context, align: "center" });
text("Decoder", 1050, 178, 180, 20, { size: 11, bold: true, color: C.rep, align: "center" });

const input = rect("Input image\n384 x 384", 95, 278, 140, 74, C.inputFill, C.input, { bold: true });
const enc = rect("MobileNetV3-Large\nbackbone", 290, 248, 200, 120, C.encFill, C.enc, { bold: true });
const aspp = rect("ASPP\nsemantic pyramid", 560, 248, 185, 120, "#E0F2FE", "#0284C7", { bold: true });
const lgc = rect("LGC\nlocal-global lesion context", 790, 248, 220, 120, C.contextFill, C.context, { bold: true });
const up = rect("Upsample\nhigh-level map", 1060, 248, 145, 120, "#F1F5F9", "#94A3B8");
const lcsf = rect("LCSF\nsemantic-boundary fusion", 890, 430, 250, 86, C.fusionFill, C.fusion, { bold: true });
const low = rect("Shortcut feature\nlow-level boundary detail", 310, 430, 190, 72, "#EEF2FF", "#6366F1");
const cat = rect("Concat", 1195, 430, 100, 86, "#F1F5F9", "#94A3B8");
const rep = rect("RepConv decoder\ntrain multi-branch\nfuse to 3x3 conv", 1340, 410, 165, 126, C.repFill, C.rep, { bold: true, fontSize: 12 });
const out = rect("Six-class mask\nbackground / leaf /\nrust / alternaria /\ngray / brown", 1335, 230, 170, 118, C.outFill, C.out, { bold: true, fontSize: 12 });

badge("1", 775, 232, C.context);
badge("2", 875, 414, C.fusion);
badge("3", 1324, 394, C.rep);
badge("4", 1275, 126, C.loss);

edge(input, enc, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
edge(enc, aspp, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
edge(aspp, lgc, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5, color: C.context });
edge(lgc, up, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
edge(up, lcsf, { exitX: 0.5, exitY: 1, entryX: 0.62, entryY: 0, color: C.fusion, points: [[1132, 395]] });
edge(enc, low, { exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0, color: "#6366F1", dashed: true, label: "low-level" });
edge(low, lcsf, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5, color: C.fusion, dashed: true, label: "boundary skip" });
edge(lcsf, cat, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
edge(cat, rep, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5, color: C.rep });
edge(rep, out, { exitX: 0.5, exitY: 0, entryX: 0.5, entryY: 1, color: C.out, points: [[1422, 370]] });
edge(out, rep, { exitX: 0.03, exitY: 1, entryX: 0.98, entryY: 0.12, color: C.loss, dashed: true, label: "CE all + Dice lesions", points: [[1300, 580], [1535, 580], [1535, 390]] });

// Inset module cards
rect("", 60, 600, 1420, 300, "#FFFFFF", C.hair, { strokeWidth: 1, arc: 14 });
text("B  Module mechanisms", 86, 620, 230, 24, { size: 15, bold: true });

rect("1  LGC after ASPP\nlocal DWConv 3x3/5x5\n+ global context gate\n+ spatial lesion gate\nresidual update", 90, 668, 300, 145, C.contextFill, C.context, { align: "left", bold: true, fontSize: 12, spacing: 12 });
rect("2  LCSF before concat\nhigh-to-low gate\nlow-to-high gate\nedge branch\nsemantic-boundary alignment", 425, 668, 300, 145, C.fusionFill, C.fusion, { align: "left", bold: true, fontSize: 12, spacing: 12 });
rect("3  RepConv decoder\ntraining: 3x3 + 1x1 + identity\ninference: fused 3x3\n2 fused blocks verified", 760, 668, 300, 145, C.repFill, C.rep, { align: "left", bold: true, fontSize: 12, spacing: 12 });
rect("4  LesionDice2\nCE supervises all 6 classes\nDice only classes 2-5\nfocus rust / alternaria / gray / brown", 1095, 668, 300, 145, C.lossFill, C.loss, { align: "left", bold: true, fontSize: 12, spacing: 12 });

// Compact evidence footer
text("C  Evidence snapshot", 86, 925, 230, 24, { size: 15, bold: true });
pill("Baseline\n71.72 / 66.58", 285, 914, 135, 52, "#F8FAFC", "#94A3B8", { bold: true });
pill("+LGC\n76.17 / 71.68", 448, 914, 135, 52, C.contextFill, C.context, { bold: true });
pill("+BalancedPrefix\n76.60 / 72.22", 612, 914, 160, 52, C.lossFill, C.loss, { bold: true });
pill("+RepConv\n76.94 / 72.63", 802, 914, 145, 52, C.repFill, C.rep, { bold: true });
pill("+LesionDice2\n77.10 / 72.83", 976, 914, 155, 52, C.outFill, C.out, { bold: true });
rect("+LGC+LCSF caveat: seed11 full/e80 is available, but seed23 is still needed before claiming a formal dual-seed ablation row.", 1180, 910, 300, 62, C.warnFill, C.warn, { bold: true, fontSize: 11 });

const diagramXml = `  <diagram name="Fig.1 Method overview" id="page-${++n}">
    <mxGraphModel dx="1600" dy="1050" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1560" pageHeight="1010" background="#FFFFFF" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        ${cells.join("\n        ")}
      </root>
    </mxGraphModel>
  </diagram>`;

const xml = `<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="Electron" agent="${agent}" version="30.0.4">
${diagramXml}
</mxfile>
`;

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.mkdirSync(path.dirname(REPO_COPY), { recursive: true });
fs.writeFileSync(TARGET, xml, "utf8");
fs.writeFileSync(REPO_COPY, xml, "utf8");

for (const packFile of [PACK, REPO_PACK]) {
  if (fs.existsSync(packFile)) {
    const oldPack = fs.readFileSync(packFile, "utf8");
    const updated = oldPack.replace(/<diagram name="Fig\.1 Method overview"[\s\S]*?<\/diagram>/, diagramXml.trim());
    fs.writeFileSync(packFile, updated, "utf8");
    console.log(`Updated first page in ${packFile}`);
  }
}

console.log(`Wrote ${TARGET}`);
console.log(`Wrote ${REPO_COPY}`);
