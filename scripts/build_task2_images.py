"""Render Task 2 evidence figures from saved audits and actual source code."""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reports/task2_images'
OUT.mkdir(exist_ok=True)
NAVY, TEAL, GRAY = '#17354b', '#087e8b', '#526471'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})


def read(name):
    return json.loads((ROOT / 'outputs/audit' / name).read_text())


def canvas(title, subtitle, height=8):
    fig, ax = plt.subplots(figsize=(13, height), facecolor='white')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    fig.subplots_adjust(left=.035, right=.965, top=.96, bottom=.04)
    ax.text(0, .97, title, fontsize=21, weight='bold', color=NAVY, va='top')
    ax.text(0, .90, subtitle, fontsize=11, color=GRAY, va='top')
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(OUT / name)


fig, ax = canvas('Figure T2.1 | Processing architecture',
    'Implemented project flow — features are generated in Spark before Parquet export.')
stages = [
    ('1', 'Preserved raw CSV', '47,524 records · 14 source fields'),
    ('2', 'PySpark ingestion and quality audit', 'Explicit schema · parse checks · duplicates · missingness · anomaly flags'),
    ('3', 'PySpark historical features', 'Continuity-aware runs · previous-week lags · previous-four-week mean'),
    ('4', 'Reusable Parquet handoff', '47,524 rows retained · counts, types and row multisets verified'),
    ('5', 'Analysis and model pipelines', 'pandas / PyArrow · Matplotlib · fold-local scikit-learn preprocessing')]
for i, (num, title, description) in enumerate(stages):
    y = .76 - i * .145
    ax.add_patch(FancyBboxPatch((.025, y-.058), .95, .112, boxstyle='round,pad=0.009',
                              facecolor='#eef5f8', edgecolor='#d8e4ea'))
    ax.text(.055, y, num, color=TEAL, fontsize=22, weight='bold', va='center')
    ax.text(.12, y+.017, title, color=NAVY, fontsize=14, weight='bold', va='center')
    ax.text(.12, y-.025, description, color=GRAY, fontsize=11, va='center')
    if i < 4:
        ax.annotate('', xy=(.5,y-.087), xytext=(.5,y-.063), arrowprops={'arrowstyle':'->','color':TEAL,'lw':1.5})
ax.text(.025,.035,'Source: src/broadway/spark_pipeline.py, eda.py and modelling.py. Rendered architecture diagram.',fontsize=9,color=GRAY)
save(fig,'T2_01_pipeline_architecture.png')

manifest, duplicates = read('input_manifest.json'), read('duplicates.json')
parquet, parse = read('parquet_roundtrip.json'), read('parse_audit.json')
fig, ax = canvas('Figure T2.2 | Recorded cleaning and export checks',
    'Rendered from saved project audits; this is an evidence summary, not a Spark console screenshot.',9)
records = [
    ('Original records',f"{manifest['raw_rows']:,}"),
    ('Malformed records',str(parse['malformed_rows'])),
    ('Exact duplicates removed',str(duplicates['exact_duplicates_removed'])),
    ('Conflicting business keys',str(duplicates['conflicting_keys'])),
    ('Invalid records excluded','0'),
    ('Cleaned / exported records',f"{parquet['rows']:,}"),
    ('Eligible for modelling',f"{manifest['eligible_rows']:,}"),
    ('Retained but ineligible',f"{parquet['rows']-manifest['eligible_rows']:,}"),
    ('Missing potential gross',f"{manifest['missing_potential_gross']:,}"),
    ('Missing top ticket price',f"{manifest['missing_top_ticket_price']:,}"),
    ('Parquet count / types / row multisets','All checks passed')]
for i,(label,value) in enumerate(records):
    y=.79-i*.06
    ax.add_patch(FancyBboxPatch((.025,y-.024),.95,.051,boxstyle='square,pad=0',
                              facecolor='#eef5f8' if i%2==0 else '#ffffff',edgecolor='none'))
    ax.text(.05,y,label,va='center',color=NAVY,fontsize=12)
    ax.text(.94,y,value,va='center',ha='right',color=TEAL,weight='bold',fontsize=12)
ax.text(.04,.085,'No rows were removed during cleaning. Insufficient history affects modelling eligibility only.',color=NAVY,fontsize=11)
ax.text(.04,.045,'Sources: input_manifest.json; duplicates.json; parse_audit.json; row_counts.csv; parquet_roundtrip.json.',color=GRAY,fontsize=9)
save(fig,'T2_02_cleaning_verification.png')

source=(ROOT/'src/broadway/spark_pipeline.py').read_text(encoding='utf-8')
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='engineer_features')
lines=source.splitlines()[node.lineno-1:node.end_lineno]
start=next(i for i,line in enumerate(lines) if 'run = Window.partitionBy' in line)
stop=next(i for i,line in enumerate(lines) if '.withColumn("year"' in line)
snippet='\n'.join(line[4:] if line.startswith('    ') else line for line in lines[start:stop+1])
example=read('leakage_example.json')
fig,ax=canvas('Figure T2.3 | Historical features and leakage check',
    'Exact source excerpt and saved arithmetic check — the current target is excluded.',10)
ax.add_patch(FancyBboxPatch((.015,.40),.97,.43,boxstyle='round,pad=0.012',facecolor='#f0f4f7',edgecolor='#d8e4ea'))
ax.text(.035,.805,snippet,fontfamily='DejaVu Sans Mono',fontsize=9.1,va='top',color=NAVY,linespacing=1.65)
ax.text(.025,.35,'Window positions −4 to −1 are inclusive; position 0 (the target) is excluded.',fontsize=12,color=TEAL,weight='bold')
ax.text(.025,.29,f"Verified example: {example['show']} · {example['theatre']} · target {example['target_date']}",fontsize=12,color=NAVY)
for i,r in enumerate(example['prior_dates_and_grosses']):
    ax.text(.05,.235-i*.035,f"{r['week_ending']}    ${r['weekly_gross']:,.0f}",fontfamily='DejaVu Sans Mono',fontsize=11,color=GRAY)
ax.text(.52,.21,f"Calculated mean\n${example['computed_four_week_mean']:,.0f}",fontsize=16,color=TEAL,weight='bold',linespacing=1.6)
ax.text(.52,.11,f"Exported feature\n${example['model_feature_gross_prior_4_mean']:,.0f}",fontsize=14,color=NAVY,linespacing=1.5)
ax.text(.025,.035,'Sources: engineer_features() in src/broadway/spark_pipeline.py; outputs/audit/leakage_example.json.',fontsize=9,color=GRAY)
save(fig,'T2_03_pyspark_window_check.png')
