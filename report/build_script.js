// Builds report/Video_script.docx from the slides' speaker notes (report/slides/script_rows.json).
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, AlignmentType } = require("docx");
const FONT = "Times New Roman";
const rows = JSON.parse(fs.readFileSync(path.join(__dirname, "slides", "script_rows.json"), "utf8"));
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const widths = [900, 1500, 6626];
const cell = (text, i, header) => new TableCell({
  borders, width: { size: widths[i], type: WidthType.DXA },
  shading: header ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
  margins: { top: 60, bottom: 60, left: 100, right: 100 },
  children: [new Paragraph({ children: [new TextRun({ text, font: FONT, size: 24, bold: header })] })],
});
const span = (who) => { const n = rows.filter((r) => r[2] === who).map((r) => r[0]); return `${Math.min(...n)} to ${Math.max(...n)}`; };
const words = (who) => rows.filter((r) => r[2] === who).reduce((a, r) => a + r[3], 0);
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 24 } } } },
  sections: [{
    properties: { page: { margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } } },
    children: [
      new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "House Price Prediction: video script", font: FONT, size: 32, bold: true })] }),
      new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: `About ${Math.round((words("Member 1") + words("Member 2")) / 125 * 10) / 10} minutes at 125 words per minute. Member 2: slides ${span("Member 2")} (${words("Member 2")} words). Member 1: slides ${span("Member 1")} (${words("Member 1")} words). The Member 1 / Member 2 labels can be swapped. Replace [Member 1], [Member 2] and [team name] before recording.`, font: FONT, size: 24, italics: true })] }),
      new Table({
        width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
        rows: [new TableRow({ tableHeader: true, children: ["Slide", "Speaker", "Script"].map((t, i) => cell(t, i, true)) }),
          ...rows.map((r) => new TableRow({ cantSplit: true, children: [String(r[0]), r[2], r[4]].map((t, i) => cell(t, i, false)) }))],
      }),
    ],
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(path.join(__dirname, "Video_script.docx"), b); console.log("wrote Video_script.docx"); });
