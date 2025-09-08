"""Top-10 product suggester using OpenAI Chat API with a deterministic fallback.

Usage: set OPENAI_API_KEY in environment and run:
    python agent.py --csv path/to/sales.csv --model gpt-4o-mini --top-k 10

The CSV should contain (at minimum): product_id, title, total_sales, avg_rating, num_reviews
Optional column: comments (text blob or sample reviews)
"""

from __future__ import annotations

import os
import json
import math
from typing import List, Dict, Any, Optional

import pandas as pd


def read_products(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    required = ['product_id', 'title', 'total_sales', 'avg_rating', 'num_reviews']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column in CSV: {c}")
    # Coerce numeric types
    df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce').fillna(0).astype(float)
    df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce').fillna(0).astype(float)
    df['num_reviews'] = pd.to_numeric(df['num_reviews'], errors='coerce').fillna(0).astype(int)
    if 'comments' not in df.columns:
        df['comments'] = ''
    return df


def local_score(row: pd.Series) -> float:
    """Deterministic fallback scoring combining sales, rating and reviews.

    Heuristic: score = avg_rating * log10(1 + total_sales) * log10(1 + num_reviews)
    This ensures products with many sales and reviews outrank single-sale 5-star items.
    """
    sales = float(row['total_sales'])
    rating = float(row['avg_rating'])
    reviews = int(row['num_reviews'])
    return rating * math.log10(1 + sales) * math.log10(1 + reviews)


def build_payload(df: pd.DataFrame, max_products: int = 200) -> List[Dict[str, Any]]:
    # Keep the top N by sales to make prompt small
    sample = df.sort_values('total_sales', ascending=False).head(max_products)
    products = []
    for _, r in sample.iterrows():
        products.append({
            'product_id': str(r['product_id']),
            'title': str(r['title']),
            'total_sales': float(r['total_sales']),
            'avg_rating': float(r['avg_rating']),
            'num_reviews': int(r['num_reviews']),
            'comments': str(r.get('comments',''))[:1000]
        })
    return products


def call_openai_rank(products: List[Dict[str, Any]], model: str = 'gpt-4o-mini', top_k: int = 10, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Call OpenAI Chat API to request a ranked top-k list.

    The model is asked to return JSON with `ranking`: a list of objects {product_id, score, reason}.
    """
    # Use environment variable only; do not hardcode API keys in source.
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        # Caller can choose to proceed without OpenAI; here we raise so callers know why openai failed
        raise RuntimeError('OPENAI_API_KEY not found in env')

    # Keep the message compact: only include necessary fields.
    products_brief = [
        {k: v for k, v in p.items() if k in ('product_id','title','total_sales','avg_rating','num_reviews')}
        for p in products
    ]

    system = (
        "You are a product-ranking assistant. Given a list of products with metrics, "
        "return the top N products by real-world desirability for buyers. Use total_sales, "
        "avg_rating and num_reviews; do NOT overvalue a 5.0 rating with only a single sale. "
        "Prefer products with substantial sales and consistent high ratings."
    )

    user = (
        f"Rank the following products and return a JSON object with a key 'ranking' containing up to {top_k} entries. "
        "Each entry must be an object with keys: product_id (string), score (number 0-100), reason (short text).\n"
        "Input products (array):\n" + json.dumps(products_brief, ensure_ascii=False)
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            temperature=0.0,
            max_tokens=800,
        )
        text = resp.choices[0].message.content
        # Try to extract JSON blob from the response
        start = text.find('{')
        if start >= 0:
            js = text[start:]
            return {'success': True, 'text': text, 'json': json.loads(js)}
        else:
            return {'success': False, 'text': text, 'error': 'no json found'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def suggest(csv_path: str, model: str = 'gpt-4o-mini', top_k: int = 10, api_key: Optional[str] = None) -> Dict[str, Any]:
    df = read_products(csv_path)
    products = build_payload(df)

    # First attempt: ask OpenAI
    openai_res = None
    try:
        openai_res = call_openai_rank(products, model=model, top_k=top_k, api_key=api_key)
    except Exception as e:
        openai_res = {'success': False, 'error': str(e)}

    # Fallback: deterministic local ranking
    df['local_score'] = df.apply(local_score, axis=1)
    local_rank = df.sort_values('local_score', ascending=False).head(top_k)
    local_ranking = [
        {'product_id': str(r['product_id']), 'title': str(r['title']), 'score': float(r['local_score'])}
        for _, r in local_rank.iterrows()
    ]

    return {
        'openai': openai_res,
        'local': local_ranking,
        'meta': {'n_products_considered': len(products)}
    }


def write_results_csv(out: Dict[str, Any], csv_path: str, out_path: str) -> None:
    """Write ranking results to CSV. Combines OpenAI ranking (if present) and local ranking.

    Columns: product_id, title, score, source, reason
    """
    df = read_products(csv_path)
    title_map = {str(r['product_id']): r['title'] for _, r in df.iterrows()}

    rows = []
    # OpenAI results first (if present and parsed)
    openai = out.get('openai') or {}
    if openai.get('success') and isinstance(openai.get('json'), dict):
        ranking = openai['json'].get('ranking') or []
        for item in ranking:
            pid = str(item.get('product_id'))
            rows.append({
                'product_id': pid,
                'title': title_map.get(pid, ''),
                'score': item.get('score'),
                'source': 'openai',
                'reason': item.get('reason', '')
            })

    # Local ranking
    for item in out.get('local', []):
        rows.append({
            'product_id': str(item.get('product_id')),
            'title': item.get('title',''),
            'score': item.get('score'),
            'source': 'local',
            'reason': ''
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Suggest top products using OpenAI and a local fallback')
    p.add_argument('--csv', required=True, help='Path to CSV with product metrics')
    p.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    p.add_argument('--top-k', type=int, default=10, help='Number of top products requested')
    p.add_argument('--api-key', default=None, help='OpenAI API key (optional, falls back to OPENAI_API_KEY)')
    p.add_argument('--out', default=None, help='Path to output CSV file (optional)')
    args = p.parse_args()

    out = suggest(args.csv, model=args.model, top_k=args.top_k, api_key=args.api_key)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out:
        try:
            write_results_csv(out, args.csv, args.out)
            print(f'Wrote ranking CSV to: {args.out}')
        except Exception as e:
            print('Failed writing CSV:', str(e))
