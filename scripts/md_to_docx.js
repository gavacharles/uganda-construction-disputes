// Build a Word document from a simple Markdown file, following the project's writing conventions
// (Times New Roman 12 pt, 1.5 line spacing, A4, 1-inch margins, live hyperlinks, page numbers).
// Supports: "# " title, "## "/"### " headings, paragraphs, "- " bullets, pipe tables, **bold**, *italic*, [text](url)
// (link text may itself be *italic*), and ![caption](image.png) figures (300-dpi PNGs, scaled to width).
// Usage: node scripts/md_to_docx.js input.md output.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ExternalHyperlink, HeadingLevel, AlignmentType, Footer,
  PageNumber, LevelFormat, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require("docx");

const FONT = "Times New Roman";
const LINE = 360; // 1.5 lines (240 = single)

// Split inline markdown into runs: links, bold, italic, plain text.
function inline(text, base = {}) {
  const out = [];
  const re = /\[((?:[^\]\[]|\[[^\]]*\])+)\]\(([^)\s]+)\)|\*\*([^*]+)\*\*|\*([^*]+)\*/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    if (m[1] !== undefined) {
      out.push(new ExternalHyperlink({
        link: m[2],
        children: inline(m[1], { ...base, style: "Hyperlink" }),
      }));
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

// Pipe table -> Word table (header row shaded and bold; widths in DXA, proportional to text length).
function table(block) {
  const rows = block.split("\n").filter((l) => !/^\s*\|?\s*:?-{3,}/.test(l))
    .map((l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const ncol = rows[0].length, total = 9026; // A4 text width at 1-inch margins
  // width ~ content length (capped), but never narrower than the longest single word needs
  const word = (j) => Math.max(...rows.map((r) => Math.max(...(r[j] || "").replace(/\*/g, "").split(/\s+/).map((w) => w.length))));
  const len = Array.from({ length: ncol }, (_, j) =>
    Math.max(Math.min(Math.max(...rows.map((r) => (r[j] || "").length)), 48), word(j) * 1.9 + 3, 6));
  const sum = len.reduce((a, b) => a + b, 0);
  const widths = len.map((x) => Math.round((total * x) / sum));
  widths[ncol - 1] += total - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0, children: r.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      borders: { top: border, bottom: border, left: border, right: border },
      shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF6" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ spacing: { line: 276, after: 0 }, children: inline(c, i === 0 ? { bold: true, size: 22 } : { size: 22 }) })],
    })) })),
  });
}

function build(md, dir) {
  const blocks = md.replace(/\r/g, "").split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
  const children = [];
  for (const b of blocks) {
    if (b.startsWith("# ")) {
      children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: inline(b.slice(2)) }));
    } else if (b.startsWith("### ")) {
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: inline(b.slice(4)) }));
    } else if (b.startsWith("## ")) {
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: inline(b.slice(3)) }));
    } else if (/^!\[/.test(b)) {
      const [, cap, file] = b.match(/^!\[([^\]]*)\]\(([^)]+)\)/);
      const img = fs.readFileSync(path.resolve(dir, file));
      const w = img.readUInt32BE(16), h = img.readUInt32BE(20); // PNG header
      const width = 600, height = Math.round((h / w) * width);
      children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: img, transformation: { width, height } })] }));
      children.push(new Paragraph({ style: "Caption", children: inline(cap) }));
    } else if (b.split("\n").every((l) => l.trim().startsWith("|"))) {
      children.push(table(b));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
    } else if (b.split("\n").every((l) => l.startsWith("- "))) {
      for (const l of b.split("\n")) children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: inline(l.slice(2)) }));
    } else {
      children.push(new Paragraph({ children: inline(b.replace(/\n/g, " ")) }));
    }
  }
  return new Document({
    styles: {
      default: { document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { line: LINE, after: 160 } } } },
      paragraphStyles: [
        { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
          run: { font: FONT, size: 34, bold: true }, paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 24, bold: true, italics: true }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 1 } },
        { id: "Caption", name: "Caption", basedOn: "Normal", next: "Normal",
          run: { font: FONT, size: 20, italics: true }, paragraph: { spacing: { before: 60, after: 240 } } },
      ],
    },
    numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) },
      children,
    }],
  });
}

const [, , input, output] = process.argv;
const doc = build(fs.readFileSync(input, "utf8"), path.dirname(path.resolve(input)));
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(output, buf); console.log(`wrote ${output}`); });
