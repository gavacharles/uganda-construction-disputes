// Build the Hove & Quill Uganda Construction Sector Report 2026 (.docx) from its Markdown source.
//
// Layout: branded cover; contents; body with running header and page numbers; the scorecard as a full-bleed page in
// its own section; chapters start on a new page; "What this means for your contract" blockquotes become shaded
// call-out boxes; pipe tables get brand-styled headers; figures are scaled to the text width with captions kept
// with their images. Body text follows the project conventions (Times New Roman 12 pt, 1.5 spacing, A4).
// Brand colours are from hovequill.com (gold #B08A33, ink #14120F, gray #57534B, stone #E7E3D8, ivory #F6F3EC).
//
// Markdown supported: front matter up to the first "---" (title, *subtitle*, byline lines, *cut-off note*);
// "## " chapter, "### " section; paragraphs; "- " bullets; "1. " numbered lists; pipe tables; "> " call-outs;
// ![caption](figure.png); **bold**, *italic*, [text](url).
// Usage: node scripts/build_sector_report.js <input.md> <output.docx> [--compact]
//   --compact: for short documents (concept notes): no contents page, chapters do not start on a new page.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ExternalHyperlink, HeadingLevel, AlignmentType, Header, Footer, PageNumber,
  LevelFormat, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, TableOfContents, TabStopType,
  PageBreak, SectionType,
} = require("docx");

const COMPACT = process.argv.includes("--compact");
const FONT = "Times New Roman";
const BRAND = { gold: "B08A33", goldDeep: "8A6B22", ink: "14120F", gray: "57534B", muted: "7D776C", stone: "E7E3D8",
  ivory: "F6F3EC" };
const LINE = 360;                       // 1.5 lines
const A4 = { w: 11906, h: 16838 };      // twips
const MARGIN = 1440;                    // 1 inch
const TEXT_W = A4.w - 2 * MARGIN;       // 9026 twips = 6.27 in
const PX_PER_TWIP = 96 / 1440;
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

// ---------- inline markdown ----------
function inline(text, base = {}) {
  const out = [];
  const re = /\[((?:[^\]\[]|\[[^\]]*\])+)\]\(([^)\s]+)\)|\*\*([^*]+)\*\*|\*([^*]+)\*/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    if (m[1] !== undefined) {
      out.push(new ExternalHyperlink({ link: m[2], children: inline(m[1], { ...base, style: "Hyperlink" }) }));
    } else if (m[3] !== undefined) {
      out.push(...inline(m[3], { ...base, bold: true }));
    } else {
      out.push(...inline(m[4], { ...base, italics: true }));
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

// ---------- blocks ----------
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }

function figure(cap, file, dir, widthTwips = TEXT_W) {
  const img = fs.readFileSync(path.resolve(dir, file));
  const { w, h } = pngSize(img);
  const width = Math.round(widthTwips * PX_PER_TWIP), height = Math.round((h / w) * width);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60, line: 240 },
      children: [new ImageRun({ type: "png", data: img, transformation: { width, height },
        altText: { title: cap.split(".")[0], description: cap, name: path.basename(file) } })] }),
    new Paragraph({ style: "Caption", children: inline(cap) }),
  ];
}

function table(block) {
  const rows = block.split("\n").filter((l) => !/^\s*\|?\s*:?-{3,}/.test(l))
    .map((l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const ncol = rows[0].length;
  const word = (j) => Math.max(...rows.map((r) => Math.max(...(r[j] || "").replace(/\*/g, "").split(/\s+/).map((w) => w.length))));
  const len = Array.from({ length: ncol }, (_, j) =>
    Math.max(Math.min(Math.max(...rows.map((r) => (r[j] || "").length)), 48), word(j) * 1.9 + 3, 8));
  const sum = len.reduce((a, b) => a + b, 0);
  const widths = len.map((x) => Math.round((TEXT_W * x) / sum));
  widths[ncol - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const line = { style: BorderStyle.SINGLE, size: 4, color: BRAND.stone };
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
    borders: { top: line, bottom: line, left: NONE, right: NONE, insideHorizontal: line, insideVertical: NONE },
    rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, cantSplit: true, children: r.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: BRAND.stone } : undefined,
      margins: { top: 70, bottom: 70, left: 100, right: 100 },
      children: [new Paragraph({ spacing: { line: 252, after: 0 },
        children: inline(c, i === 0 ? { bold: true, size: 19, color: BRAND.ink } : { size: 19 }) })],
    })) })),
  });
}

function callout(lines) {
  const body = lines.map((l) => l.replace(/^>\s?/, ""));
  const title = body[0].replace(/\*\*/g, "");
  const kids = [new Paragraph({ spacing: { after: 100, line: 276 },
    children: [new TextRun({ text: title.toUpperCase(), font: FONT, bold: true, size: 19, color: BRAND.goldDeep,
      characterSpacing: 20 })] })];
  for (const l of body.slice(1)) {
    const t = l.replace(/^-\s+/, "");
    kids.push(new Paragraph({ numbering: l.startsWith("- ") ? { reference: "callout", level: 0 } : undefined,
      spacing: { after: 80, line: 288 }, children: inline(t, { size: 21 }) }));
  }
  return [new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W],
    borders: { top: NONE, bottom: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE,
      left: { style: BorderStyle.SINGLE, size: 36, color: BRAND.gold } },
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: TEXT_W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, color: "auto", fill: BRAND.ivory },
      margins: { top: 180, bottom: 140, left: 260, right: 220 }, children: kids })] })],
  }), spacer()];
}
function spacer() { const p = new Paragraph({ spacing: { after: 160 }, children: [] }); p.isSpacer = true; return p; }

let listInstance = 0;
function blocks(md, dir) {
  const out = [];
  const parts = md.replace(/\r/g, "").split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
  for (const b of parts) {
    if (b === "---") continue;
    if (b.startsWith("## ")) {
      while (out.length && out[out.length - 1].isSpacer) out.pop();   // avoid a blank page before a chapter
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: !COMPACT,
        spacing: COMPACT && out.length ? { before: 480 } : undefined, children: inline(b.slice(3)) }));
    } else if (b.startsWith("### ")) {
      out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: inline(b.slice(4)) }));
    } else if (/^!\[/.test(b)) {
      const [, cap, file] = b.match(/^!\[([^\]]*)\]\(([^)]+)\)/);
      if (/scorecard_page/.test(file)) { out.push({ scorecard: { cap, file } }); continue; }
      out.push(...figure(cap, file, dir));
    } else if (b.split("\n").every((l) => l.trim().startsWith("|"))) {
      out.push(table(b), spacer());
    } else if (b.split("\n").every((l) => l.startsWith(">"))) {
      out.push(...callout(b.split("\n")));
    } else if (b.split("\n").every((l) => /^\d+\.\s/.test(l))) {
      listInstance += 1;   // each numbered list restarts at 1
      for (const l of b.split("\n")) {
        out.push(new Paragraph({ numbering: { reference: "findings", level: 0, instance: listInstance },
          spacing: { after: 140, line: 320 },
          children: inline(l.replace(/^\d+\.\s+/, "")) }));
      }
    } else if (b.split("\n").every((l) => l.startsWith("- "))) {
      for (const l of b.split("\n")) {
        out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 },
          children: inline(l.slice(2)) }));
      }
    } else {
      out.push(new Paragraph({ children: inline(b.replace(/\n/g, " ")) }));
    }
  }
  return out;
}

// ---------- cover ----------
function cover(front, dir) {
  const lines = front.split("\n").map((l) => l.trim()).filter(Boolean);
  const title = lines.find((l) => l.startsWith("# ")).slice(2);
  const [main, sub] = title.split(":").map((s) => s.trim());
  const italic = lines.filter((l) => /^\*[^*].*\*$/.test(l)).map((l) => l.slice(1, -1));
  const byline = lines.filter((l) => !l.startsWith("# ") && !/^\*[^*].*\*$/.test(l)).map((l) => l.replace(/\*\*/g, ""));
  const mark = fs.readFileSync(path.resolve(dir, "brand/hq-mark.png"));
  const run = (text, o = {}) => new TextRun({ text, font: FONT, ...o });
  return [
    new Paragraph({ spacing: { after: 360 }, children: [new ImageRun({ type: "png", data: mark,
      transformation: { width: 84, height: 84 }, altText: { title: "Hove & Quill", description: "Hove & Quill mark", name: "mark" } })] }),
    new Paragraph({ spacing: { after: 40 }, children: [run("HOVE & QUILL", { size: 30, bold: true, color: BRAND.ink, characterSpacing: 120 })] }),
    new Paragraph({ spacing: { after: 1800 }, children: [run("CONSTRUCTION PRACTICE", { size: 18, color: BRAND.goldDeep, characterSpacing: 80 })] }),
    new Paragraph({ spacing: { after: 120, line: 240 }, border: { top: { style: BorderStyle.SINGLE, size: 18, color: BRAND.gold, space: 18 } },
      children: [run(sub ? sub.toUpperCase() : "", { size: 22, color: BRAND.goldDeep, characterSpacing: 60, bold: true })] }),
    new Paragraph({ spacing: { after: 240, line: 240 }, children: [run(main, { size: 64, bold: true, color: BRAND.ink })] }),
    new Paragraph({ spacing: { after: 1600, line: 300 }, children: [run(italic[0] || "", { size: 28, italics: true, color: BRAND.gray })] }),
    ...byline.slice(1).map((l, i) => new Paragraph({ spacing: { after: 60 },
      children: [run(l, { size: i === 0 ? 24 : 22, bold: i === 0, color: i === 0 ? BRAND.ink : BRAND.gray })] })),
    new Paragraph({ spacing: { before: 1400, line: 264 }, border: { top: { style: BorderStyle.SINGLE, size: 4, color: BRAND.stone, space: 8 } },
      children: [run(italic[1] || "", { size: 17, color: BRAND.muted })] }),
  ];
}

// ---------- header / footer ----------
const tabs = [{ type: TabStopType.RIGHT, position: TEXT_W }];
const small = (text, o = {}) => new TextRun({ text, font: FONT, size: 17, color: BRAND.muted, ...o });
const runningHeader = (short) => new Header({ children: [new Paragraph({ tabStops: tabs,
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BRAND.stone, space: 4 } },
  children: [small("HOVE & QUILL", { characterSpacing: 60, color: BRAND.goldDeep, bold: true }), small(`\t${short}`)] })] });
const runningFooter = (status) => new Footer({ children: [new Paragraph({ tabStops: tabs,
  children: [small(status), small("\t"), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 17, color: BRAND.ink })] })] });
const empty = { header: new Header({ children: [new Paragraph({ children: [] })] }),
  footer: new Footer({ children: [new Paragraph({ children: [] })] }) };

// ---------- document ----------
function build(md, dir) {
  const [front, ...rest] = md.replace(/\r/g, "").split(/\n---\n/);
  const body = blocks(rest.join("\n---\n"), dir);
  const cut = body.findIndex((b) => b && b.scorecard);
  const before = cut >= 0 ? body.slice(0, cut) : body, after = cut >= 0 ? body.slice(cut + 1) : [];
  const title = (front.match(/^# (.*)$/m) || [, "Report"])[1];
  const short = title.split(":")[0];
  const status = (front.match(/·\s*(Draft[^\n*]*)/) || [, ""])[1].trim() || "Hove & Quill";
  const page = (m = MARGIN) => ({ size: { width: A4.w, height: A4.h }, margin: { top: m, bottom: m, left: m, right: m,
    header: 560, footer: 560 } });
  const hf = { headers: { default: runningHeader(short) }, footers: { default: runningFooter(status) } };

  const sections = [
    { properties: { page: page() }, headers: empty.header && { default: empty.header }, footers: { default: empty.footer },
      children: cover(front, dir) },
    { properties: { type: SectionType.NEXT_PAGE, page: { ...page(), pageNumbers: { start: 1 } } }, ...hf,
      children: COMPACT ? before : [
        new Paragraph({ spacing: { after: 280, line: 276 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: BRAND.gold, space: 6 } },
          children: [new TextRun({ text: "Contents", font: FONT, size: 36, bold: true, color: BRAND.ink })] }),
        new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        ...before] },
  ];
  if (cut >= 0) {
    const sc = body[cut].scorecard;
    const m = 500, w = A4.w - 2 * m;
    sections.push({ properties: { type: SectionType.NEXT_PAGE, page: page(m) }, headers: { default: empty.header },
      footers: { default: empty.footer }, children: [figure(sc.cap, sc.file, dir, w)[0]] });
    sections.push({ properties: { type: SectionType.NEXT_PAGE, page: page() }, ...hf, children: after });
  }

  return new Document({
    creator: "Hove & Quill", title, description: "Uganda construction sector report 2026",
    features: { updateFields: true },
    styles: {
      default: { document: { run: { font: FONT, size: 24, color: "1F1D1A" }, paragraph: { spacing: { line: LINE, after: 160 } } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 36, bold: true, color: BRAND.ink },
          paragraph: { spacing: { before: 0, after: 280, line: 276 }, keepNext: true, outlineLevel: 0,
            border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: BRAND.gold, space: 6 } } } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 26, bold: true, color: BRAND.goldDeep },
          paragraph: { spacing: { before: 320, after: 120, line: 276 }, keepNext: true, outlineLevel: 1 } },
        { id: "Caption", name: "Caption", basedOn: "Normal", next: "Normal",
          run: { font: FONT, size: 18, italics: true, color: BRAND.gray },
          paragraph: { spacing: { before: 40, after: 280, line: 252 } } },
      ],
      characterStyles: [{ id: "Hyperlink", name: "Hyperlink", basedOn: "DefaultParagraphFont",
        run: { color: "2B67A8", underline: { type: "single", color: "2B67A8" } } }],
    },
    numbering: { config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { run: { color: BRAND.gold }, paragraph: { indent: { left: 540, hanging: 300 } } } }] },
      { reference: "callout", levels: [{ level: 0, format: LevelFormat.BULLET, text: "▪", alignment: AlignmentType.LEFT,
        style: { run: { color: BRAND.gold }, paragraph: { indent: { left: 360, hanging: 260 } } } }] },
      { reference: "findings", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1", alignment: AlignmentType.LEFT,
        style: { run: { color: BRAND.gold, bold: true, size: 28 }, paragraph: { indent: { left: 620, hanging: 620 } } } }] },
    ] },
    sections,
  });
}

const [input, output] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
if (!input || !output) { console.error("Usage: node scripts/build_sector_report.js input.md output.docx"); process.exit(1); }
const doc = build(fs.readFileSync(input, "utf8"), path.dirname(path.resolve(input)));
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(output, buf); console.log(`wrote ${output} (${(buf.length / 1e6).toFixed(1)} MB)`); });
