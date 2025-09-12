"""Business analysis assistant for model prediction results.

This script reads a CSV exported from the `product_predictions` MongoDB collection
with columns like:
    _id, asin, true_rating, pred_rating, ratings_total, n_texts, sample_text,
    abs_err, signed_err, true_bin, pred_bin, run_ts, model, max_len, batch_sz,
    brand, type

It computes quantitative performance / error summaries and (optionally) asks an
OpenAI model to generate a concise business-oriented analysis and
recommendations (inventory, marketing, product / data quality improvements).

Environment:
    Set OPENAI_API_KEY for AI-generated narrative (optional). If absent, a
    deterministic local narrative is produced.

Example:
    python agent.py --csv product_predictions.csv --model gpt-4o-mini

Outputs JSON to stdout containing:
    summary: numeric + tabular aggregates
    ai_summary: (optional) model-generated business analysis
    local_summary: fallback deterministic human-readable text
"""

from __future__ import annotations

import os
import json
import math
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np


########## Data Loading & Summarization #########################################################

def read_predictions(csv_path: str) -> pd.DataFrame:
    """Load prediction CSV and coerce numeric columns safely.

    Missing expected columns are created with NaNs so downstream metrics still run.
    """
    df = pd.read_csv(csv_path)
    numeric_cols = [
        'true_rating','pred_rating','abs_err','signed_err','ratings_total',
        'true_bin','pred_bin'
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Text convenience columns
    for c in ['brand','type','sample_text']:
        if c not in df.columns:
            df[c] = ''
    return df


def summarize_predictions(df: pd.DataFrame, max_categories: int = 10) -> Dict[str, Any]:
    n = len(df)
    if n == 0:
        return {'n': 0, 'error': 'empty dataframe'}

    mae = float(df['abs_err'].mean()) if 'abs_err' in df else float((df['true_rating'] - df['pred_rating']).abs().mean())
    rmse = float(math.sqrt(((df['true_rating'] - df['pred_rating']) ** 2).mean())) if 'true_rating' in df and 'pred_rating' in df else float('nan')
    bias = float(df['signed_err'].mean()) if 'signed_err' in df else float((df['pred_rating'] - df['true_rating']).mean())

    # Bin accuracy
    bin_accuracy = float((df['true_bin'] == df['pred_bin']).mean()) if 'true_bin' in df and 'pred_bin' in df else float('nan')

    # Confusion matrix (sparse) limited to 10x10; convert to nested dict
    confusion = {}
    if 'true_bin' in df and 'pred_bin' in df:
        ct = pd.crosstab(df['true_bin'], df['pred_bin'])
        if ct.shape[0] <= 15 and ct.shape[1] <= 15:  # keep payload small
            confusion = {str(r): {str(c): int(ct.loc[r, c]) for c in ct.columns} for r in ct.index}

    # Brand performance
    brand_grp = (
        df.groupby('brand')
          .agg(count=('brand','size'), avg_abs_err=('abs_err','mean'))
          .sort_values('count', ascending=False)
          .head(max_categories)
    )
    brand_perf = [
        {
            'brand': b,
            'count': int(r['count']) if not pd.isna(r['count']) else None,
            'avg_abs_err': float(r['avg_abs_err']) if not pd.isna(r['avg_abs_err']) else None,
        }
        for b, r in brand_grp.iterrows()
    ]

    type_grp = (
        df.groupby('type')
          .agg(count=('type','size'), avg_abs_err=('abs_err','mean'))
          .sort_values('count', ascending=False)
          .head(max_categories)
    )
    type_perf = [
        {
            'type': b,
            'count': int(r['count']) if not pd.isna(r['count']) else None,
            'avg_abs_err': float(r['avg_abs_err']) if not pd.isna(r['avg_abs_err']) else None,
        }
        for b, r in type_grp.iterrows()
    ]

    # Worst absolute errors (top 5)
    worst = []
    if 'abs_err' in df:
        cols_for_worst = [c for c in ['asin','brand','type','true_rating','pred_rating','abs_err','ratings_total','sample_text'] if c in df.columns]
        worst_rows = df.sort_values('abs_err', ascending=False).head(5)[cols_for_worst]
        for _, r in worst_rows.iterrows():
            worst.append({k: (None if pd.isna(r[k]) else (str(r[k]) if k=='sample_text' else r[k])) for k in cols_for_worst})

    return {
        'n_records': n,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'bias_mean_signed_err': bias,
            'bin_accuracy': bin_accuracy,
            'mean_true_rating': float(df['true_rating'].mean()) if 'true_rating' in df else None,
            'mean_pred_rating': float(df['pred_rating'].mean()) if 'pred_rating' in df else None,
        },
        'brand_perf': brand_perf,
        'type_perf': type_perf,
        'confusion': confusion,
        'worst_predictions': worst,
    }


########## Configuration (discouraged hardcoded key) ###########################################

# WARNING: Hardcoding secrets is insecure. This is only for a disposable demo environment.
# For production use environment variables or a secret manager.
HARD_CODED_OPENAI_KEY = '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'  # <-- optionally place a short-lived demo key here

########## AI Narrative #########################################################################

def call_openai_analysis(summary: Dict[str, Any], model: str = 'gpt-4o-mini', api_key: Optional[str] = None) -> Dict[str, Any]:
    api_key = api_key or os.getenv('OPENAI_API_KEY') or HARD_CODED_OPENAI_KEY
    if not api_key:
        return {'success': False, 'error': 'missing OPENAI_API_KEY'}

    # Keep payload small; stringify key parts only
    compact = {
        'metrics': summary.get('metrics'),
        'brand_perf': summary.get('brand_perf'),
        'type_perf': summary.get('type_perf'),
        'worst_predictions': summary.get('worst_predictions'),
        'n_records': summary.get('n_records'),
    }

    system = (
        "You are a senior business analyst. You receive model prediction performance "
        "metrics for product rating predictions (true vs predicted star ratings). Provide: \n"
        "1. Performance overview (accuracy, error, bias).\n"
        "2. Reliability risks (where model under/over performs: brands, types, outliers).\n"
        "3. Business implications (inventory, marketing, customer satisfaction).\n"
        "4. Concrete next actions (data collection, model tuning, product focus).\n"
        "Be concise (<350 words) and use bullet lists. Avoid repeating raw numbers excessively; interpret them."
    )
    user = (
        "Here are aggregated metrics from a product rating prediction evaluation (JSON):\n" + json.dumps(compact, ensure_ascii=False)
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            temperature=0.2,
            max_tokens=700,
        )
        text = resp.choices[0].message.content.strip()
        return {'success': True, 'text': text}
    except Exception as e:
        return {'success': False, 'error': str(e)}


########## Local Fallback Narrative #############################################################

def build_local_narrative(summary: Dict[str, Any]) -> str:
    m = summary.get('metrics', {})
    mae = m.get('mae')
    rmse = m.get('rmse')
    bias = m.get('bias_mean_signed_err')
    acc = m.get('bin_accuracy')
    n = summary.get('n_records')

    def fmt(x):
        return 'n/a' if x is None or (isinstance(x, float) and (np.isnan(x))) else f"{x:.3f}" if isinstance(x, float) else str(x)

    lines = []
    lines.append(f"Dataset size: {n} records. MAE={fmt(mae)}, RMSE={fmt(rmse)}, Bin accuracy={fmt(acc)}, Bias={fmt(bias)}.")
    if abs(bias) > 0.2:
        lines.append("Noticeable bias: predictions systematically " + ("over" if bias>0 else "under") + "-estimate true ratings.")
    elif not np.isnan(bias):
        lines.append("Bias is modest; mean signed error near zero.")

    # Brand commentary
    brands = summary.get('brand_perf', [])[:5]
    if brands:
        worst = sorted([b for b in brands if b.get('avg_abs_err') is not None], key=lambda x: x['avg_abs_err'], reverse=True)[:1]
        best = sorted([b for b in brands if b.get('avg_abs_err') is not None], key=lambda x: x['avg_abs_err'])[:1]
        if best:
            lines.append(f"Best brand (error): {best[0]['brand']} avg_abs_err={fmt(best[0]['avg_abs_err'])}.")
        if worst:
            lines.append(f"Most challenging brand: {worst[0]['brand']} avg_abs_err={fmt(worst[0]['avg_abs_err'])}.")

    # Outliers
    worst_preds = summary.get('worst_predictions', [])[:3]
    if worst_preds:
        lines.append("Top error outliers (asin -> abs_err): " + ", ".join(f"{w.get('asin')}->{fmt(float(w.get('abs_err')))}" for w in worst_preds if w.get('abs_err') is not None))

    lines.append("Recommended next steps: 1) Gather more training data for high-error brands/types. 2) Investigate text patterns in outliers. 3) Calibrate model or ensemble for underperforming segments. 4) Use high-confidence segments for marketing claims; monitor others.")
    return "\n".join(lines)


########## Table Explanations ###################################################################

def build_table_explanations(summary: Dict[str, Any]) -> Dict[str, str]:
    """Return 3-5 line plain-text explanations for each table + overall summary."""
    m = summary.get('metrics', {})
    lines_summary = []
    lines_summary.append(
        f"Model shows MAE {m.get('mae'):.3f} and RMSE {m.get('rmse'):.3f}, indicating typical absolute error under ~{m.get('mae'):.2f} stars." if m.get('mae') is not None else "Overall metrics unavailable." )
    if m.get('bias_mean_signed_err') is not None:
        bias = m.get('bias_mean_signed_err')
        if abs(bias) < 0.1:
            lines_summary.append("Bias is minimal; predictions centered around true ratings.")
        else:
            lines_summary.append(f"Bias of {bias:.3f} suggests systematic {'under' if bias<0 else 'over'} prediction.")
    if m.get('bin_accuracy') is not None:
        lines_summary.append(f"Bin accuracy {m.get('bin_accuracy'):.2%} shows how often coarse rating buckets match.")
    lines_summary.append("Focus improvements on reducing large outlier errors influencing RMSE.")

    brand_perf = summary.get('brand_perf', [])
    brand_lines = []
    if brand_perf:
        worst = max(brand_perf, key=lambda x: (x.get('avg_abs_err') or -1))
        best = min(brand_perf, key=lambda x: (x.get('avg_abs_err') or 999))
        brand_lines.append(f"Lowest error brand: {best['brand']} ({best['avg_abs_err']:.3f}). Highest: {worst['brand']} ({worst['avg_abs_err']:.3f}).")
        brand_lines.append("Gap suggests segment-specific behavior; consider stratified retraining.")
        high_err = [b for b in brand_perf if b.get('avg_abs_err') and b['avg_abs_err']> (m.get('mae',0)*1.2)]
        if high_err:
            brand_lines.append("Brands above MAE*1.2: " + ", ".join(b['brand'] for b in high_err[:5]))
        brand_lines.append("Prioritize data enrichment or feature engineering for high-error brands.")
    else:
        brand_lines.append("No brand data available.")

    type_perf = summary.get('type_perf', [])
    type_lines = []
    if type_perf:
        worst_t = max(type_perf, key=lambda x: (x.get('avg_abs_err') or -1))
        best_t = min(type_perf, key=lambda x: (x.get('avg_abs_err') or 999))
        type_lines.append(f"Best type: {best_t['type']} ({best_t['avg_abs_err']:.3f}); hardest: {worst_t['type']} ({worst_t['avg_abs_err']:.3f}).")
        type_lines.append("Differences may reflect distinct text vocabularies or rating distributions.")
        type_lines.append("Consider separate model heads or domain-specific fine-tuning if disparity persists.")
    else:
        type_lines.append("No type-level aggregates available.")

    worst_preds = summary.get('worst_predictions', [])
    worst_lines = []
    if worst_preds:
        top = worst_preds[0]
        worst_lines.append(f"Largest absolute error {top.get('abs_err'):.3f} on ASIN {top.get('asin')} (true {top.get('true_rating')}, pred {top.get('pred_rating')}).")
        worst_lines.append("Outliers often have atypical wording or sparse context; inspect text samples.")
        worst_lines.append("Mitigate via data augmentation or model calibration (e.g., isotonic).")
    else:
        worst_lines.append("No outlier rows captured.")

    conf = summary.get('confusion') or {}
    conf_lines = []
    if conf:
        # derive simple stats
        correct = 0
        total = 0
        for tb, preds in conf.items():
            for pb, cnt in preds.items():
                total += cnt
                if tb == pb:
                    correct += cnt
        if total:
            conf_lines.append(f"Overall bin accuracy {correct/total:.2%}; most errors likely near adjacent bins.")
        conf_lines.append("Confusion concentrated where true ratings border decision thresholds.")
    else:
        conf_lines.append("No confusion matrix available.")

    return {
        'overall': " ".join(lines_summary[:5]),
        'brand_perf': " ".join(brand_lines[:5]),
        'type_perf': " ".join(type_lines[:5]),
        'worst_predictions': " ".join(worst_lines[:5]),
        'confusion': " ".join(conf_lines[:5]),
    }


########## Public API ###########################################################################

def analyze(csv_path: str, model: str = 'gpt-4o-mini', api_key: Optional[str] = None) -> Dict[str, Any]:
    df = read_predictions(csv_path)
    summary = summarize_predictions(df)
    local_text = build_local_narrative(summary)
    ai = call_openai_analysis(summary, model=model, api_key=api_key)
    explanations = build_table_explanations(summary)
    out: Dict[str, Any] = {
        'summary': summary,
        'local_summary': local_text,
        'table_explanations': explanations,
    }
    if ai.get('success'):
        out['ai_summary'] = ai.get('text')
    else:
        out['ai_summary_error'] = ai.get('error')
    return out


########## Reporting ###########################################################################

def write_csv_reports(out: Dict[str, Any], prefix: str) -> List[str]:
    """Write structured portions of analysis output into multiple CSV files.

    Files produced (prefix=<P>):
        <P>_metrics.csv
        <P>_brand_perf.csv
        <P>_type_perf.csv
        <P>_worst_predictions.csv
        <P>_confusion.csv (long form: true_bin,pred_bin,count)
    Returns list of written file paths.
    """
    written: List[str] = []
    summary = out.get('summary', {})
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)

    metrics_path = f"{prefix}_metrics.csv"
    pd.DataFrame([summary.get('metrics', {})]).to_csv(metrics_path, index=False)
    written.append(metrics_path)

    brand = pd.DataFrame(summary.get('brand_perf', []))
    if not brand.empty:
        brand.to_csv(f"{prefix}_brand_perf.csv", index=False)
        written.append(f"{prefix}_brand_perf.csv")

    typ = pd.DataFrame(summary.get('type_perf', []))
    if not typ.empty:
        typ.to_csv(f"{prefix}_type_perf.csv", index=False)
        written.append(f"{prefix}_type_perf.csv")

    worst = pd.DataFrame(summary.get('worst_predictions', []))
    if not worst.empty:
        worst.to_csv(f"{prefix}_worst_predictions.csv", index=False)
        written.append(f"{prefix}_worst_predictions.csv")

    confusion = summary.get('confusion') or {}
    if confusion:
        rows = []
        for tb, preds in confusion.items():
            for pb, cnt in preds.items():
                rows.append({'true_bin': tb, 'pred_bin': pb, 'count': cnt})
        pd.DataFrame(rows).to_csv(f"{prefix}_confusion.csv", index=False)
        written.append(f"{prefix}_confusion.csv")
    return written


def write_html_report(out: Dict[str, Any], html_path: str) -> str:
    """Generate a standalone HTML report with narrative and tables (no external assets)."""
    summary = out.get('summary', {})
    metrics = summary.get('metrics', {})

    def table(df: pd.DataFrame, title: str) -> str:
        if df is None or df.empty:
            return ''
        return f"<h3>{title}</h3>" + df.to_html(index=False, escape=True, classes='tbl', border=0)

    brand_df = pd.DataFrame(summary.get('brand_perf', []))
    type_df = pd.DataFrame(summary.get('type_perf', []))
    worst_df = pd.DataFrame(summary.get('worst_predictions', []))
    # Confusion matrix to wide format
    conf = summary.get('confusion') or {}
    conf_df = None
    if conf:
        conf_df = pd.DataFrame(conf).fillna(0).astype(int)
        conf_df.index.name = 'true_bin'
        conf_df.reset_index(inplace=True)

    ai_text = out.get('ai_summary') or 'AI narrative unavailable.'
    local_text = out.get('local_summary','')
    expl = out.get('table_explanations', {})
    def expl_div(key: str) -> str:
        txt = expl.get(key)
        return f"<p class='explanation'>{txt}</p>" if txt else ''

    # Use double braces to escape in f-string for CSS blocks
    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <title>Prediction Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; line-height:1.4; }}
    h1,h2,h3 {{ color:#233; }}
    .narrative {{ background:#f5f7fa; padding:1rem; border-radius:6px; }}
    table.tbl {{ border-collapse:collapse; margin:1rem 0; font-size:0.9rem; }}
    table.tbl th, table.tbl td {{ border:1px solid #ccc; padding:4px 6px; text-align:left; }}
    .metrics span {{ display:inline-block; margin-right:1rem; }}
    footer {{ margin-top:2rem; font-size:0.75rem; color:#666; }}
  </style>
</head>
<body>
  <h1>Product Rating Prediction Analysis</h1>
  <section class='metrics'>
    <h2>Key Metrics</h2>
    {''.join(f"<span><strong>{k}</strong>: {v}</span>" for k,v in metrics.items())}
  </section>
  <section class='narrative'>
    <h2>AI Narrative</h2>
    <pre style='white-space:pre-wrap'>{ai_text}</pre>
    <h2>Deterministic Summary</h2>
    <pre style='white-space:pre-wrap'>{local_text}</pre>
  </section>
    <section>
        <h2>Overall Summary</h2>
        {expl_div('overall')}
    </section>
    {table(brand_df, 'Brand Performance')}{expl_div('brand_perf')}
    {table(type_df, 'Type Performance')}{expl_div('type_perf')}
    {table(worst_df, 'Worst Absolute Errors')}{expl_div('worst_predictions')}
    {table(conf_df, 'Confusion Matrix (True vs Pred Bin)')}{expl_div('confusion')}
  <footer>Generated by agent.py</footer>
</body>
</html>"""

    os.makedirs(os.path.dirname(html_path) or '.', exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return html_path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Business analysis of product rating prediction results')
    p.add_argument('--csv', required=True, help='Path to product_predictions CSV')
    p.add_argument('--model', default='gpt-4o-mini', help='OpenAI model for narrative (optional)')
    p.add_argument('--api-key', default=None, help='Explicit OpenAI API key (else OPENAI_API_KEY env)')
    p.add_argument('--report-prefix', default=None, help='Prefix path for writing CSV report components')
    p.add_argument('--html', default=None, help='Path to write an HTML summary report')
    args = p.parse_args()

    result = analyze(args.csv, model=args.model, api_key=args.api_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report_prefix:
        written = write_csv_reports(result, args.report_prefix)
        print('Wrote CSV components:', ', '.join(written))
    if args.html:
        path = write_html_report(result, args.html)
        print('Wrote HTML report:', path)
