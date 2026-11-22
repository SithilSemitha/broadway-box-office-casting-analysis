"""Create an insertable Word appendix from five exact project source excerpts."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/report_deps'))
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


def extract(relative, symbol):
    source = (ROOT / relative).read_text(encoding='utf-8')
    tree = ast.parse(source)
    if symbol == 'SCHEMA':
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets))
    else:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol)
    lines = source.splitlines()
    return '\n'.join(lines[node.lineno-1:node.end_lineno]), node.lineno, node.end_lineno


def build():
    spark = 'src/broadway/spark_pipeline.py'
    model = 'src/broadway/modelling.py'
    entries = [
        ('Explicit PySpark schema', spark, 'SCHEMA',
         'Task 2: the declared schema assigns expected types to all 14 source fields. Typed ingestion and separate parsing checks make malformed input detectable.'),
        ('Historical features and four-week eligibility', spark, 'engineer_features',
         'Task 2: continuity-based runs prevent lags crossing gaps. The window from −4 to −1 includes four earlier rows and excludes the current target. Eligibility requires four prior observations.'),
        ('Parquet export and roundtrip verification', spark, None,
         'Task 2: this excerpt from preprocess() verifies the exported row count, field types and complete row multisets in both directions. The comparison preserves duplicate multiplicities.'),
        ('Fold-local modelling pipelines', model, 'build_pipeline',
         'Task 4: the pipeline applies numeric imputation and categorical encoding during fitting. Ridge also scales numeric inputs. Fitting each pipeline on training rows keeps learned transformations within that fold.'),
        ('Chronological training and validation split', model, 'chronological_fold',
         'Task 4: date masks place training observations before the validation year. Empty or overlapping splits are rejected, and complete target weeks remain on the same side of the split.')]
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.8)
    section.left_margin = section.right_margin = Cm(1.5)
    normal = doc.styles['Normal']
    normal.font.name, normal.font.size = 'Calibri', Pt(10)
    normal.paragraph_format.space_after = Pt(6)
    for name in ['Heading 1', 'Heading 2']:
        doc.styles[name].font.name = 'Calibri'
        doc.styles[name].font.color.rgb = RGBColor.from_string('17354B')
    doc.add_heading('Appendix A — Selected implementation code', 1)
    doc.add_paragraph('The following five listings are extracted directly from the delivered project source. They support the engineering and evaluation discussion in Tasks 2 and 4. Imports, configuration constants and surrounding functions remain available in the named modules; these excerpts are not standalone programs.')
    records = []
    for i,(title,relative,symbol,description) in enumerate(entries,1):
        if i > 1:
            doc.add_page_break()
        if symbol:
            code, first, last = extract(relative,symbol)
        else:
            lines = (ROOT / relative).read_text(encoding='utf-8').splitlines()
            first = next(i+1 for i,line in enumerate(lines) if 'frame.orderBy(*KEYS).coalesce(1).write' in line)
            last = next(i for i,line in enumerate(lines) if 'bounds = raw.agg' in line)
            code = '\n'.join(lines[first-1:last])
        doc.add_heading(f'A.{i} {title}',2)
        doc.add_paragraph(description)
        doc.add_paragraph(f'Listing A.{i}. {title}. Source: {relative}, lines {first}–{last}.', 'Caption')
        for line in code.splitlines():
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.line_spacing = 1
            paragraph.paragraph_format.widow_control = False
            run = paragraph.add_run(line or ' ')
            run.font.name, run.font.size = 'Consolas',Pt(7.5)
            props = paragraph._p.get_or_add_pPr()
            shade = OxmlElement('w:shd')
            shade.set(qn('w:fill'),'F1F5F8')
            props.append(shade)
        records.append({'listing':f'A.{i}','source':relative,'start_line':first,'end_line':last,
                        'excerpt_sha256':hashlib.sha256(code.encode()).hexdigest()})
    doc.core_properties.title = 'Appendix A — Selected implementation code'
    doc.core_properties.subject = 'Five source excerpts supporting Tasks 2 and 4'
    out = ROOT / 'outputs/reports/Broadway_CW_Code_Appendix.docx'
    doc.save(out)
    with ZipFile(out) as archive:
        assert archive.testzip() is None
    record={'document':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
            'listings':records,'rendering':'DOCX package checked; Word pagination requires review.'}
    out.with_name('code_appendix_build_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(out)
    print('Five exact source excerpts saved; portrait A4 with editable code.')


if __name__ == '__main__':
    build()
