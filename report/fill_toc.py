"""Write the table of contents into a report .docx so it shows without updating fields.

docx-js leaves the TOC field empty. This script renders the document with LibreOffice,
finds the page of every Heading 1 and Heading 2, and puts those entries (with dot leaders
and page numbers) inside the TOC field. Word can still refresh them with Update Field.

Usage (system python3, which has LibreOffice's uno module):
    python3 report/fill_toc.py report/House_Prices_Report.docx
"""
import html
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

TAB_POS = 11906 - 2 * 1080  # A4 width minus the two 0.75-inch margins, in twips


def render_pdf(docx: Path, pdf: Path) -> None:
    """Open the file in headless LibreOffice, update its indexes and export a PDF."""
    import uno
    from com.sun.star.beans import PropertyValue

    def pv(name, value):
        p = PropertyValue()
        p.Name, p.Value = name, value
        return p

    office = subprocess.Popen(["soffice", "--headless", "--invisible", "--norestore",
                               "--accept=socket,host=localhost,port=2002;urp;"])
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
        for _ in range(60):
            try:
                ctx = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
                break
            except Exception:
                time.sleep(0.5)
        desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(str(docx.resolve())), "_blank", 0,
                                           (pv("Hidden", True),))
        indexes = doc.getDocumentIndexes()
        for i in range(indexes.getCount()):
            indexes.getByIndex(i).update()
        doc.storeToURL(uno.systemPathToFileUrl(str(pdf.resolve())), (pv("FilterName", "writer_pdf_Export"),))
        doc.close(True)
        try:
            desktop.terminate()
        except Exception:
            pass
    finally:
        office.wait(timeout=60)


def headings(document_xml: str) -> list[tuple[int, str]]:
    """(level, text) of every Heading1/Heading2 paragraph, in document order."""
    found = []
    for para in re.findall(r"<w:p\b.*?</w:p>", document_xml, flags=re.S):
        style = re.search(r'<w:pStyle w:val="Heading([12])"/>', para)
        if style:
            text = "".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", para, flags=re.S))
            found.append((int(style.group(1)), html.unescape(text)))
    return found


def page_numbers(pdf: Path, titles: list[str]) -> list[int]:
    """Page of each title, searching forward page by page (the contents page itself is skipped)."""
    n_pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                                               text=True).stdout).group(1))
    pages = [subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"], capture_output=True,
                            text=True).stdout.splitlines() for p in range(1, n_pages + 1)]
    contents_page = next(i for i, lines in enumerate(pages) if any(l.strip() == "Contents" for l in lines))
    result, page = [], 0
    for title in titles:
        while True:
            if page != contents_page and any(l.strip() == title for l in pages[page]):
                result.append(page + 1)
                break
            page += 1
            if page >= n_pages:
                raise SystemExit(f"heading not found in the PDF: {title!r}")
    return result


def entry(level: int, title: str, page: int, prefix: str = "", suffix: str = "") -> str:
    return (f'<w:p><w:pPr><w:pStyle w:val="TOC{level}"/><w:tabs><w:tab w:val="right" w:leader="dot" '
            f'w:pos="{TAB_POS}"/></w:tabs></w:pPr>{prefix}<w:r><w:t xml:space="preserve">{html.escape(title)}'
            f'</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>{page}</w:t></w:r>{suffix}</w:p>')


def main(path: str) -> None:
    docx = Path(path)
    with zipfile.ZipFile(docx) as z:
        files = {name: z.read(name) for name in z.namelist()}
    xml = files["word/document.xml"].decode("utf8")

    field = re.search(r'<w:p><w:r><w:fldChar w:fldCharType="begin"[^>]*/><w:instrText xml:space="preserve">'
                      r'TOC [^<]*</w:instrText><w:fldChar w:fldCharType="separate"/></w:r></w:p>'
                      r'(.*?)<w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>', xml, flags=re.S)
    if not field:
        raise SystemExit("TOC field not found")
    begin = re.search(r'<w:r><w:fldChar w:fldCharType="begin".*?w:fldCharType="separate"/></w:r>',
                      field.group(0), flags=re.S).group(0)

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "render.pdf"
        render_pdf(docx, pdf)
        heads = headings(xml)
        pages = page_numbers(pdf, [t for _, t in heads])

    rows = [entry(level, title, page) for (level, title), page in zip(heads, pages)]
    rows[0] = entry(*heads[0], pages[0], prefix=begin)
    rows[-1] = entry(*heads[-1], pages[-1], suffix='<w:r><w:fldChar w:fldCharType="end"/></w:r>')
    files["word/document.xml"] = (xml[:field.start()] + "".join(rows) + xml[field.end():]).encode("utf8")

    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    for (level, title), page in zip(heads, pages):
        print(f"{'    ' * (level - 1)}{title} ... {page}")


if __name__ == "__main__":
    main(sys.argv[1])
