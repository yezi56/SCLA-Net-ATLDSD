const fs = require("fs");
const path = require("path");

const OUT_DIR = "D:\\Pictures";
const PACK_FILE = path.join(OUT_DIR, "ATLDSD_paper_figures.drawio");
const ARCH_FILE = path.join(OUT_DIR, "ATLDSD_final_module_architecture.drawio");

const mxAgent =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) draw.io/29.6.6 Chrome/144.0.7559.236 Electron/40.8.4 Safari/537.36";

const palette = {
  ink: "#111827",
  muted: "#64748B",
  line: "#475569",
  blueFill: "#DBEAFE",
  blue: "#2563EB",
  skyFill: "#E0F2FE",
  sky: "#0284C7",
  tealFill: "#CCFBF1",
  teal: "#0F766E",
  greenFill: "#DCFCE7",
  green: "#16A34A",
  amberFill: "#FEF3C7",
  amber: "#D97706",
  orangeFill: "#FFEDD5",
  orange: "#EA580C",
  roseFill: "#FFE4E6",
  rose: "#E11D48",
  violetFill: "#EDE9FE",
  violet: "#7C3AED",
  slateFill: "#F8FAFC",
  slate: "#CBD5E1",
  grayFill: "#F1F5F9",
  gray: "#94A3B8",
};

let uid = 1000;
function id(prefix = "atldsd") {
  uid += 1;
  return `${prefix}-${uid}`;
}

function esc(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/\n/g, "&#xa;");
}

function style(parts) {
  return Object.entries(parts)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${v}`)
    .join(";") + ";";
}

function baseText(extra = {}) {
  return style({
    text: "",
    strokeColor: "none",
    fillColor: "none",
    html: 1,
    fontFamily: "Helvetica",
    fontColor: extra.fontColor || palette.ink,
    fontSize: extra.fontSize || 12,
    fontStyle: extra.fontStyle,
    align: extra.align || "center",
    verticalAlign: extra.verticalAlign || "middle",
    whiteSpace: "wrap",
    rounded: 0,
  });
}

function boxStyle(fill, stroke, opts = {}) {
  return style({
    rounded: 1,
    whiteSpace: "wrap",
    html: 1,
    arcSize: opts.arcSize || 8,
    absoluteArcSize: 1,
    fillColor: fill,
    strokeColor: stroke,
    fontColor: opts.fontColor || palette.ink,
    fontSize: opts.fontSize || 12,
    fontFamily: "Helvetica",
    fontStyle: opts.bold ? 1 : undefined,
    strokeWidth: opts.strokeWidth || 1.4,
    spacing: opts.spacing || 8,
    shadow: 0,
  });
}

function pillStyle(fill, stroke, opts = {}) {
  return style({
    rounded: 1,
    whiteSpace: "wrap",
    html: 1,
    arcSize: 50,
    absoluteArcSize: 1,
    fillColor: fill,
    strokeColor: stroke,
    fontColor: opts.fontColor || palette.ink,
    fontSize: opts.fontSize || 11,
    fontFamily: "Helvetica",
    fontStyle: opts.bold ? 1 : undefined,
    strokeWidth: opts.strokeWidth || 1.2,
    spacing: 5,
    shadow: 0,
  });
}

function edgeStyle(color = palette.line, opts = {}) {
  return style({
    edgeStyle: opts.edgeStyle || "none",
    rounded: opts.rounded || 0,
    orthogonalLoop: 1,
    jettySize: "auto",
    html: 1,
    strokeColor: color,
    strokeWidth: opts.strokeWidth || 2,
    endArrow: opts.endArrow || "block",
    endFill: opts.endFill === undefined ? 1 : opts.endFill,
    startArrow: opts.startArrow,
    startFill: opts.startFill,
    dashed: opts.dashed ? 1 : undefined,
    dashPattern: opts.dashPattern,
    exitX: opts.exitX,
    exitY: opts.exitY,
    entryX: opts.entryX,
    entryY: opts.entryY,
    exitDx: 0,
    exitDy: 0,
    entryDx: 0,
    entryDy: 0,
  });
}

function page(name, width, height, build) {
  const cells = [];
  const api = {
    cells,
    text(value, x, y, w, h, opts = {}) {
      const cid = id("t");
      cells.push(
        `<mxCell id="${cid}" parent="1" value="${esc(value)}" vertex="1" style="${baseText(opts)}"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry" /></mxCell>`
      );
      return cid;
    },
    box(value, x, y, w, h, fill, stroke, opts = {}) {
      const cid = id("b");
      cells.push(
        `<mxCell id="${cid}" parent="1" value="${esc(value)}" vertex="1" style="${boxStyle(fill, stroke, opts)}"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry" /></mxCell>`
      );
      return cid;
    },
    pill(value, x, y, w, h, fill, stroke, opts = {}) {
      const cid = id("p");
      cells.push(
        `<mxCell id="${cid}" parent="1" value="${esc(value)}" vertex="1" style="${pillStyle(fill, stroke, opts)}"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry" /></mxCell>`
      );
      return cid;
    },
    ellipse(value, x, y, w, h, fill, stroke, opts = {}) {
      const cid = id("e");
      const st = style({
        ellipse: "",
        whiteSpace: "wrap",
        html: 1,
        fillColor: fill,
        strokeColor: stroke,
        fontColor: opts.fontColor || palette.ink,
        fontSize: opts.fontSize || 12,
        fontFamily: "Helvetica",
        fontStyle: opts.bold ? 1 : undefined,
        strokeWidth: opts.strokeWidth || 1.4,
        spacing: 6,
      });
      cells.push(
        `<mxCell id="${cid}" parent="1" value="${esc(value)}" vertex="1" style="${st}"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry" /></mxCell>`
      );
      return cid;
    },
    edge(source, target, opts = {}) {
      const cid = id("g");
      const value = opts.label ? ` value="${esc(opts.label)}"` : "";
      let geom = `<mxGeometry relative="1" as="geometry"`;
      if (opts.points && opts.points.length) {
        geom += `><Array as="points">${opts.points
          .map((p) => `<mxPoint x="${p[0]}" y="${p[1]}" />`)
          .join("")}</Array></mxGeometry>`;
      } else {
        geom += ` />`;
      }
      cells.push(
        `<mxCell id="${cid}"${value} parent="1" source="${source}" target="${target}" edge="1" style="${edgeStyle(opts.color, opts)}">${geom}</mxCell>`
      );
      return cid;
    },
    line(x1, y1, x2, y2, color = palette.slate, opts = {}) {
      const a = this.ellipse("", x1 - 1, y1 - 1, 2, 2, "none", "none");
      const b = this.ellipse("", x2 - 1, y2 - 1, 2, 2, "none", "none");
      return this.edge(a, b, {
        color,
        strokeWidth: opts.strokeWidth || 1,
        endArrow: "none",
        endFill: 0,
      });
    },
  };
  build(api);
  return `<diagram name="${esc(name)}" id="${id("page")}"><mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="${width}" pageHeight="${height}" background="#ffffff" math="0" shadow="0"><root><mxCell id="0" /><mxCell id="1" parent="0" />${cells.join("")}</root></mxGraphModel></diagram>`;
}

function buildArchitecturePage() {
  return page("Fig.1 Method overview", 1700, 1060, (d) => {
    d.text("Lesion-focused DeepLabV3+ for ATLDSD semantic segmentation", 255, 35, 850, 34, { fontSize: 24, fontStyle: 1 });
    d.text("Single-stage six-class mask prediction: context modeling, cross-scale fusion, deployable decoder refinement, lesion-focused optimization", 255, 73, 850, 24, { fontSize: 12, fontColor: palette.muted });
    d.pill("dual-seed avg\nmIoU 77.10", 1160, 36, 125, 46, palette.greenFill, palette.green, { bold: true });
    d.pill("FG mIoU\n72.83", 1298, 36, 105, 46, palette.skyFill, palette.sky, { bold: true });
    d.pill("lesion Dice\nclasses 2-5", 1416, 36, 120, 46, palette.orangeFill, palette.orange, { bold: true });

    const input = d.box("Input image\n384 x 384", 75, 210, 135, 76, palette.slateFill, palette.gray, { bold: true });
    const enc = d.box("MobileNetV3-Large\nencoder", 270, 190, 185, 116, palette.blueFill, palette.blue, { bold: true });
    const low = d.box("Low-level feature\nboundary detail", 295, 385, 150, 70, "#EEF2FF", "#6366F1");
    const aspp = d.box("ASPP\nhigh-level semantics", 520, 190, 178, 116, palette.skyFill, palette.sky, { bold: true });
    const lgc = d.box("LGC\nlocal-global lesion context", 760, 190, 205, 116, palette.tealFill, palette.teal, { bold: true });
    const up = d.box("Upsample", 1028, 210, 132, 76, palette.grayFill, palette.gray);
    const lcsf = d.box("LCSF\nsemantic-boundary fusion", 945, 385, 215, 86, palette.violetFill, palette.violet, { bold: true });
    const cat = d.box("Concat", 1218, 390, 110, 76, palette.grayFill, palette.gray);
    const dec = d.box("RepConv decoder\ntrain: multi-branch\ndeploy: fused 3x3", 1380, 370, 185, 116, palette.amberFill, palette.amber, { bold: true });
    const out = d.box("Six-class mask\nbackground / leaf / 4 lesions", 1485, 205, 155, 90, palette.greenFill, palette.green, { bold: true });

    d.edge(input, enc, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5, color: palette.line });
    d.edge(enc, aspp, { exitX: 1, exitY: 0.42, entryX: 0, entryY: 0.42, color: palette.line });
    d.edge(aspp, lgc, { exitX: 1, exitY: 0.42, entryX: 0, entryY: 0.42, color: palette.teal });
    d.edge(lgc, up, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5, color: palette.line });
    d.edge(up, lcsf, { exitX: 0.5, exitY: 1, entryX: 0.54, entryY: 0, color: palette.violet, points: [[1095, 335]] });
    d.edge(enc, low, { exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0, color: "#6366F1" });
    d.edge(low, lcsf, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5, color: palette.violet });
    d.edge(lcsf, cat, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5, color: palette.line });
    d.edge(cat, dec, { exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5, color: palette.amber });
    d.edge(dec, out, { exitX: 0.55, exitY: 0, entryX: 0.55, entryY: 1, color: palette.green, points: [[1482, 315]] });

    d.text("where it sits in code", 92, 585, 180, 24, { fontSize: 15, fontStyle: 1, align: "left" });
    d.box("After ASPP\nx = aspp(x) -> attention_aspp(x) -> LGC(x)", 92, 625, 315, 74, palette.tealFill, palette.teal);
    d.box("Before decoder concat\nLCSF(high, low) -> concat -> cat_conv", 450, 625, 315, 74, palette.violetFill, palette.violet);
    d.box("Decoder conv replacement\nnormal conv -> RepConvBlock; fuse_for_deploy()", 808, 625, 340, 74, palette.amberFill, palette.amber);
    d.box("Optimization\nCE for all six classes + Dice only for lesion classes 2-5", 1190, 625, 365, 74, palette.orangeFill, palette.orange);

    d.text("Core evidence", 92, 760, 150, 24, { fontSize: 15, fontStyle: 1, align: "left" });
    d.pill("Baseline\n71.72 / 66.58", 92, 805, 140, 54, palette.slateFill, palette.gray, { bold: true });
    d.pill("+LGC\n76.17 / 71.68", 275, 805, 140, 54, palette.tealFill, palette.teal, { bold: true });
    d.pill("+BalancedPrefix\n76.60 / 72.22", 458, 805, 170, 54, palette.orangeFill, palette.orange, { bold: true });
    d.pill("+RepConv\n76.94 / 72.63", 672, 805, 150, 54, palette.amberFill, palette.amber, { bold: true });
    d.pill("+LesionDice2\n77.10 / 72.83", 866, 805, 165, 54, palette.greenFill, palette.green, { bold: true });
    d.box("+LGC+LCSF formal row still needs seed23\ncurrent evidence is partial; do not overclaim in paper", 1080, 792, 475, 80, palette.roseFill, palette.rose, { bold: true });
  });
}

function buildProblemPage() {
  return page("Fig.2 Problem and claim", 1500, 930, (d) => {
    d.text("Problem definition: lightweight single-stage multi-class apple leaf lesion segmentation", 180, 35, 1120, 34, { fontSize: 23, fontStyle: 1 });
    const img = d.box("RGB apple leaf image", 95, 210, 190, 95, palette.slateFill, palette.gray, { bold: true });
    const mask = d.box("Six-class semantic mask\nbackground / leaf / rust / alternaria / gray / brown", 420, 190, 280, 135, palette.greenFill, palette.green, { bold: true });
    const sev = d.box("Disease-aware outputs\nlesion area, category IoU, severity cue", 835, 205, 255, 105, palette.skyFill, palette.sky, { bold: true });
    const paper = d.box("Paper claim\nbetter lesion-class segmentation without detection-style branches", 1200, 195, 220, 125, palette.violetFill, palette.violet, { bold: true });
    d.edge(img, mask, { color: palette.line });
    d.edge(mask, sev, { color: palette.line });
    d.edge(sev, paper, { color: palette.line });

    d.text("Why the baseline struggles", 95, 430, 260, 24, { fontSize: 16, fontStyle: 1, align: "left" });
    d.box("Small lesion regions\nclass pixels are sparse", 95, 475, 245, 76, palette.roseFill, palette.rose, { bold: true });
    d.box("Fuzzy lesion boundary\nleaf texture and disease edge mix", 95, 580, 245, 76, palette.orangeFill, palette.orange, { bold: true });
    d.box("Imbalanced disease classes\ngray/brown/alternaria instability", 95, 685, 245, 76, palette.amberFill, palette.amber, { bold: true });

    d.text("Design response", 485, 430, 220, 24, { fontSize: 16, fontStyle: 1, align: "left" });
    const r1 = d.box("LGC after ASPP\nrecover lesion context at high semantic level", 485, 475, 320, 76, palette.tealFill, palette.teal, { bold: true });
    const r2 = d.box("LCSF before concat\nalign high-level lesion semantics with low-level edges", 485, 580, 320, 76, palette.violetFill, palette.violet, { bold: true });
    const r3 = d.box("RepConv + LesionDice2\ntrain robustly, deploy with fused decoder conv", 485, 685, 320, 76, palette.greenFill, palette.green, { bold: true });

    d.text("Evidence standards used in the paper", 955, 430, 335, 24, { fontSize: 16, fontStyle: 1, align: "left" });
    d.box("Formal result uses dual-seed average\ncurrent mainline: 77.10 mIoU / 72.83 FG mIoU", 955, 475, 365, 76, palette.greenFill, palette.green, { bold: true });
    d.box("Four lesion IoUs are reported separately\nrust 84.44, alternaria 58.92, gray 68.27, brown 56.95", 955, 580, 365, 76, palette.skyFill, palette.sky, { bold: true });
    d.box("Detection-style components are excluded\nno bbox, anchors, NMS, YOLO branch, or detection head", 955, 685, 365, 76, palette.roseFill, palette.rose, { bold: true });

    d.edge(r1, r2, { color: palette.violet, exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0 });
    d.edge(r2, r3, { color: palette.green, exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0 });
  });
}

function buildLgcPage() {
  return page("Fig.3 LGC module detail", 1500, 950, (d) => {
    d.text("Local-Global Lesion Context (LGC): high-level context refinement after ASPP", 170, 35, 1120, 34, { fontSize: 23, fontStyle: 1 });
    const x = d.box("ASPP feature X", 100, 330, 160, 82, palette.skyFill, palette.sky, { bold: true });
    const local = d.box("Local texture path\nDWConv 3x3 + DWConv 5x5", 410, 160, 260, 90, palette.tealFill, palette.teal, { bold: true });
    const global = d.box("Global context gate\nGAP -> 1x1 -> sigmoid", 410, 330, 260, 90, palette.blueFill, palette.blue, { bold: true });
    const spatial = d.box("Spatial lesion gate\n1x1 -> sigmoid", 410, 500, 260, 90, palette.violetFill, palette.violet, { bold: true });
    const fuse = d.box("Context fusion\nlocal * global * spatial", 825, 330, 230, 90, palette.greenFill, palette.green, { bold: true });
    const res = d.box("Residual update\nY = X + alpha * Context", 1210, 330, 210, 90, palette.amberFill, palette.amber, { bold: true });
    d.edge(x, local, { color: palette.teal, exitX: 1, exitY: 0.18, entryX: 0, entryY: 0.5, points: [[320, 200]] });
    d.edge(x, global, { color: palette.blue, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
    d.edge(x, spatial, { color: palette.violet, exitX: 1, exitY: 0.82, entryX: 0, entryY: 0.5, points: [[320, 545]] });
    d.edge(local, fuse, { color: palette.teal, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.18, points: [[755, 205]] });
    d.edge(global, fuse, { color: palette.blue, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
    d.edge(spatial, fuse, { color: palette.violet, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.82, points: [[755, 545]] });
    d.edge(fuse, res, { color: palette.green, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
    d.edge(x, res, { color: palette.gray, dashed: true, exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 1, points: [[180, 735], [1315, 735]] });
    d.text("Rationale", 100, 705, 120, 24, { fontSize: 16, fontStyle: 1, align: "left" });
    d.box("Disease spots are small and textural; LGC adds local receptive fields while preserving global disease context. The residual path keeps ASPP semantics stable.", 100, 750, 1320, 82, palette.slateFill, palette.gray, { fontSize: 13 });
  });
}

function buildLcsfPage() {
  return page("Fig.4 LCSF module detail", 1500, 960, (d) => {
    d.text("Lesion-Aware Cross-Scale Fusion (LCSF): semantic-boundary alignment before decoder concat", 105, 35, 1290, 34, { fontSize: 23, fontStyle: 1 });
    const high = d.box("High-level lesion semantics\nfrom LGC / ASPP", 95, 230, 230, 88, palette.tealFill, palette.teal, { bold: true });
    const low = d.box("Low-level boundary detail\nfrom encoder shortcut", 95, 530, 230, 88, "#EEF2FF", "#6366F1", { bold: true });
    const hg = d.box("High-to-low gate\nsemantic attention for boundary channels", 505, 205, 285, 82, palette.violetFill, palette.violet, { bold: true });
    const lg = d.box("Low-to-high gate\nboundary cue for lesion semantics", 505, 345, 285, 82, palette.skyFill, palette.sky, { bold: true });
    const edge = d.box("Edge branch\nlow-level local contrast", 505, 530, 285, 82, palette.amberFill, palette.amber, { bold: true });
    const fusion = d.box("LCSF output\nrefined high + refined low", 995, 350, 265, 105, palette.greenFill, palette.green, { bold: true });
    const concat = d.box("Decoder concat\nthen RepConv decoder", 995, 585, 265, 82, palette.grayFill, palette.gray, { bold: true });
    d.edge(high, hg, { color: palette.violet, exitX: 1, exitY: 0.35, entryX: 0, entryY: 0.5 });
    d.edge(high, lg, { color: palette.sky, exitX: 1, exitY: 0.65, entryX: 0, entryY: 0.45, points: [[410, 380]] });
    d.edge(low, hg, { color: palette.violet, dashed: true, exitX: 1, exitY: 0.35, entryX: 0, entryY: 0.8, points: [[410, 560], [410, 270]] });
    d.edge(low, edge, { color: palette.amber, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
    d.edge(hg, fusion, { color: palette.violet, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.25, points: [[900, 246]] });
    d.edge(lg, fusion, { color: palette.sky, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.52 });
    d.edge(edge, fusion, { color: palette.amber, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.82, points: [[900, 572]] });
    d.edge(fusion, concat, { color: palette.green, exitX: 0.5, exitY: 1, entryX: 0.5, entryY: 0 });
    d.text("Rationale", 95, 750, 120, 24, { fontSize: 16, fontStyle: 1, align: "left" });
    d.box("LCSF is placed before decoder concatenation, where high-level disease semantics and low-level leaf-edge details first meet. This is the most direct place to reduce fuzzy lesion-boundary confusion.", 95, 795, 1165, 82, palette.slateFill, palette.gray, { fontSize: 13 });
  });
}

function buildRepLossPage() {
  return page("Fig.5 RepConv and LesionDice2", 1500, 960, (d) => {
    d.text("Deployable decoder refinement and lesion-focused optimization", 180, 35, 1120, 34, { fontSize: 23, fontStyle: 1 });
    d.text("RepConv decoder", 105, 140, 280, 26, { fontSize: 16, fontStyle: 1, align: "left" });
    const train = d.box("Training block\n3x3 Conv+BN\n1x1 Conv+BN\nidentity BN", 105, 205, 230, 145, palette.amberFill, palette.amber, { bold: true });
    const fuse = d.box("Reparameterization\nfuse BN + pad 1x1\nsum equivalent kernels", 510, 230, 260, 105, palette.orangeFill, palette.orange, { bold: true });
    const deploy = d.box("Deploy block\nsingle 3x3 Conv", 945, 245, 210, 75, palette.greenFill, palette.green, { bold: true });
    const logits = d.box("Six-class logits", 1250, 245, 150, 75, palette.skyFill, palette.sky, { bold: true });
    d.edge(train, fuse, { color: palette.orange });
    d.edge(fuse, deploy, { color: palette.green });
    d.edge(deploy, logits, { color: palette.line });
    d.box("Deploy check\n2 fused blocks; params after fusion 12.14M; FLOPs 38.34G; CUDA1000 fused FPS 118.82; speedup 1.05x", 105, 400, 1295, 78, palette.slateFill, palette.gray, { fontSize: 13 });

    d.text("LesionDice2 objective", 105, 575, 310, 26, { fontSize: 16, fontStyle: 1, align: "left" });
    const ce = d.box("Cross-entropy\nall 6 classes", 105, 650, 210, 88, palette.blueFill, palette.blue, { bold: true });
    const dice = d.box("Dice loss\nlesion classes only\nrust / alternaria / gray / brown", 485, 630, 270, 128, palette.roseFill, palette.rose, { bold: true });
    const total = d.box("Total loss\nCE(all) + Dice(classes 2-5)", 930, 650, 270, 88, palette.greenFill, palette.green, { bold: true });
    const update = d.box("Optimization focus\nboost small lesion recall without discarding leaf/background supervision", 1050, 800, 350, 78, palette.violetFill, palette.violet, { bold: true });
    d.edge(ce, total, { color: palette.blue });
    d.edge(dice, total, { color: palette.rose });
    d.edge(total, update, { color: palette.green, exitX: 0.72, exitY: 1, entryX: 0.45, entryY: 0, points: [[1110, 775]] });
  });
}

function buildResultsPage() {
  return page("Fig.6 Ablation evidence board", 1600, 1010, (d) => {
    d.text("Ablation evidence: formal gains and current gap", 260, 35, 1080, 34, { fontSize: 23, fontStyle: 1 });
    const methods = [
      ["Baseline", 71.72, 66.58, palette.slateFill, palette.gray],
      ["+LGC", 76.17, 71.68, palette.tealFill, palette.teal],
      ["+BalancedPrefix", 76.60, 72.22, palette.orangeFill, palette.orange],
      ["+RepConv", 76.94, 72.63, palette.amberFill, palette.amber],
      ["+LesionDice2", 77.10, 72.83, palette.greenFill, palette.green],
    ];
    let prev = null;
    methods.forEach((m, i) => {
      const x = 85 + i * 292;
      const n = d.box(`${m[0]}\nmIoU ${m[1].toFixed(2)}\nFG ${m[2].toFixed(2)}`, x, 165, 205, 112, m[3], m[4], { bold: true });
      if (prev) d.edge(prev, n, { color: palette.line, exitX: 1, exitY: 0.5, entryX: 0, entryY: 0.5 });
      prev = n;
    });
    d.box("Formal mainline gain over baseline\nmIoU +5.38 / FG mIoU +6.25", 85, 340, 350, 82, palette.greenFill, palette.green, { bold: true });
    d.box("Best single seed remains historical RepConv seed23\n77.21 / 72.96; label as best single-seed only", 480, 340, 420, 82, palette.amberFill, palette.amber, { bold: true });
    d.box("+LGC+LCSF status\npartial full/e80 evidence: seed11 76.69 / 72.31; seed23 still required", 945, 340, 470, 82, palette.roseFill, palette.rose, { bold: true });

    d.text("Four lesion-class IoU under current mainline", 85, 520, 410, 26, { fontSize: 16, fontStyle: 1, align: "left" });
    const bars = [
      ["Rust", 84.44, palette.green],
      ["Alternaria", 58.92, palette.violet],
      ["Gray", 68.27, palette.sky],
      ["Brown", 56.95, palette.orange],
    ];
    bars.forEach((b, i) => {
      const y = 595 + i * 76;
      d.text(b[0], 90, y, 130, 30, { align: "right", fontSize: 13, fontStyle: 1 });
      d.box("", 240, y + 5, Math.round(820 * b[1] / 100), 20, b[2], b[2], { strokeWidth: 0.8 });
      d.text(`${b[1].toFixed(2)}`, 1080, y, 90, 30, { align: "left", fontSize: 13, fontStyle: 1 });
    });
    d.box("LesionDice2 delta vs RepConv\nmIoU +0.16; FG +0.20; brown +1.02; alternaria +0.35", 1200, 580, 310, 135, palette.greenFill, palette.green, { bold: true });
    d.box("Caveat\nrust -0.16 and gray -0.08 vs RepConv, so claim should emphasize overall lesion average and brown/alternaria stability.", 1200, 755, 310, 105, palette.amberFill, palette.amber, { bold: true });
  });
}

function buildFigurePlanPage() {
  return page("Fig.7 Paper figure checklist", 1600, 1030, (d) => {
    d.text("Recommended figure set for the ATLDSD paper", 260, 35, 1080, 34, { fontSize: 23, fontStyle: 1 });
    const rows = [
      ["Figure 1", "Problem and method overview", "Show six-class single-stage segmentation and the final pipeline", "Use Fig.1 + Fig.2 pages"],
      ["Figure 2", "LGC and LCSF module details", "Explain why context and boundary fusion address small fuzzy lesions", "Use Fig.3 + Fig.4 pages"],
      ["Figure 3", "RepConv and LesionDice2", "Support deployability and lesion-focused optimization", "Use Fig.5 page"],
      ["Figure 4", "Ablation progression", "Baseline -> LGC -> BalancedPrefix -> RepConv -> LesionDice2", "Existing PNG/PDF plus Fig.6 board"],
      ["Figure 5", "Four lesion-class IoU", "Report rust, alternaria, gray, brown separately", "fig_paper_lesion_iou_comparison"],
      ["Figure 6", "Complexity and deploy fusion", "Params, FLOPs, FPS, fused equivalence", "deploy_fused_summary + tradeoff plot"],
      ["Figure 7", "Qualitative success/failure cases", "Show input, GT, baseline, ours, error map", "paper_qualitative_cases PNG/PDF"],
    ];
    d.box("Claim carried by the whole figure set\nA lightweight single-stage model improves small, fuzzy, imbalanced lesion segmentation while staying deployable after RepConv fusion.", 95, 120, 1410, 72, palette.greenFill, palette.green, { bold: true });
    const x = [95, 240, 590, 1040];
    const w = [120, 310, 400, 375];
    d.box("ID", x[0], 245, w[0], 46, palette.slateFill, palette.gray, { bold: true });
    d.box("Panel", x[1], 245, w[1], 46, palette.slateFill, palette.gray, { bold: true });
    d.box("Purpose", x[2], 245, w[2], 46, palette.slateFill, palette.gray, { bold: true });
    d.box("Source / status", x[3], 245, w[3], 46, palette.slateFill, palette.gray, { bold: true });
    rows.forEach((r, i) => {
      const y = 305 + i * 88;
      const fill = i % 2 === 0 ? "#FFFFFF" : palette.slateFill;
      d.box(r[0], x[0], y, w[0], 64, fill, palette.slate, { bold: true });
      d.box(r[1], x[1], y, w[1], 64, fill, palette.slate);
      d.box(r[2], x[2], y, w[2], 64, fill, palette.slate);
      d.box(r[3], x[3], y, w[3], 64, fill, palette.slate);
    });
    d.box("Do not overclaim\n+LGC+LCSF needs seed23 before being used as a formal dual-seed ablation row.", 95, 930, 1410, 56, palette.roseFill, palette.rose, { bold: true });
  });
}

function mxfile(diagrams) {
  return `<?xml version="1.0" encoding="UTF-8"?>\n<mxfile host="Electron" agent="${mxAgent}" version="29.6.6">\n${diagrams.join("\n")}\n</mxfile>\n`;
}

const diagrams = [
  buildArchitecturePage(),
  buildProblemPage(),
  buildLgcPage(),
  buildLcsfPage(),
  buildRepLossPage(),
  buildResultsPage(),
  buildFigurePlanPage(),
];

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(PACK_FILE, mxfile(diagrams), "utf8");
fs.writeFileSync(ARCH_FILE, mxfile([diagrams[0]]), "utf8");
console.log(`Wrote ${PACK_FILE}`);
console.log(`Wrote ${ARCH_FILE}`);
