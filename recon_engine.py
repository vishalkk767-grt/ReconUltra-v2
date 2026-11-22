"""
recon_engine.py  -  Jarvis Ultra matching engine (Purchase Register / Books vs GSTR-2B)
------------------------------------------------------------------------------------
No Streamlit dependency, so it can be tested on its own:

    books_n, _ = normalize(books_df, books_map)
    gst_n,   _ = normalize(gstr_df,  gstr_map)
    res        = reconcile(books_n, gst_n, tol=2.0, max_gap=2, cutoff='2026-04-01')
    xlsx_bytes = build_workbook(res, cdnr_df=None, settings={...})

Matching stages (a row is used only once):
  1. GSTIN + normalised invoice no            -> Exact / Normalized
     (extra Books rows of the same invoice with same amounts -> Duplicate Invoice)
  0. Books GSTIN blank                         -> invoice no + amounts
  2. Same GSTIN, amounts within tolerance, date gap <= max_gap
                                               -> Near Match / Amount Fallback (inv no differs)
  3. Same invoice no, amounts within tolerance, GSTIN differs -> GSTIN Mismatch
  F. (optional) fuzzy invoice no, same GSTIN, date gap <= max_gap -> Fuzzy (verify)

Status given to every paired invoice (first rule that applies):
  Previous Invoice  -> 2B invoice date is before the cutoff date
  ITC Not Claimed   -> Books has no GST, 2B has GST
  Inv No Mismatch   -> paired by stage 2 / F
  GSTIN Mismatch    -> paired by stage 3
  Value Mismatch    -> taxable / IGST / CGST / SGST differs by more than tolerance (head-wise)
  Date Mismatch     -> everything matches, only date differs
  Matched           -> everything matches
"""
import re
import warnings
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO

import numpy as np
import pandas as pd

STATUS_ORDER = ['Matched', 'Date Mismatch', 'Inv No Mismatch', 'GSTIN Mismatch',
                'Duplicate Invoice', 'Value Mismatch']

PAIR_COLS = ['Invoice Date (Books)', 'Invoice Date (2B)', 'Status', 'Match Type',
             'GSTIN (Books)', 'GSTIN (2B)', 'Party Name', 'Invoice No (Books)', 'Invoice No (2B)',
             'Taxable (Books)', 'Taxable (2B)', 'Taxable Diff (Books-2B)',
             'IGST (Books)', 'IGST (2B)', 'IGST Diff (Books-2B)',
             'CGST (Books)', 'CGST (2B)', 'CGST Diff (Books-2B)',
             'SGST (Books)', 'SGST (2B)', 'SGST Diff (Books-2B)',
             'Total GST (Books)', 'Total GST (2B)', 'GST Diff (Books-2B)', 'Remark']
MISS_COLS = ['Invoice Date', 'GSTIN', 'Party Name', 'Invoice No',
             'Taxable', 'IGST', 'CGST', 'SGST', 'Total GST', 'Remark']
CDNR_COLS = ['Note Date', 'GSTIN', 'Party Name', 'Note No', 'Note Type',
             'Taxable', 'IGST', 'CGST', 'SGST', 'Total GST', 'Remark']


# ----------------------------------------------------------------------------
# Cleaning helpers
# ----------------------------------------------------------------------------
def _isnull(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT


def clean_gstin(v):
    if _isnull(v):
        return ''
    s = str(v).strip().upper().replace(' ', '').replace('\xa0', '')
    return '' if s in ('NAN', 'NONE', 'NA') else s


def clean_amount(v):
    try:
        if _isnull(v):
            return 0.0
        if isinstance(v, str):
            v = v.replace(',', '').replace('₹', '').strip()
            if v == '' or v.upper() in ('NA', 'NAN', '-'):
                return 0.0
        return round(float(v), 2)
    except Exception:
        return 0.0


def clean_text(v):
    if _isnull(v):
        return ''
    s = str(v).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    return '' if s.upper() == 'NAN' else s


def inv_key(v):
    """Normalise invoice no: ignore case, spaces and / - . _ ; strip leading zeros of every part.
       '26-27/0110' == '26-27/110' == '2627110'. Prefixes (INV, BILL, B ...) are NOT removed."""
    s = clean_text(v).upper().replace('\xa0', '')
    parts = [p.lstrip('0') or '0' for p in re.split(r'[^A-Z0-9]+', s) if p]
    return ''.join(parts)


_DATE_FORMATS = ('%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%Y-%m-%d', '%d-%b-%Y', '%d-%b-%y',
                 '%d/%m/%y', '%d %b %Y')


def parse_date(v):
    if _isnull(v):
        return pd.NaT
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.Timestamp(v).normalize()
    if isinstance(v, (int, float, np.integer, np.floating)):
        try:
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=float(v))
        except Exception:
            return pd.NaT
    s = str(v).strip()
    if s == '':
        return pd.NaT
    for f in _DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, f))
        except ValueError:
            pass
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pd.to_datetime(s, dayfirst=True, errors='coerce')


def _col(df, m, key):
    c = m.get(key)
    return df[c] if c and c in df.columns else None


def normalize(df, m):
    """Return (normalised frame, number of ignored junk rows).
       Rows with neither GSTIN nor invoice no (blank / grand-total rows) are ignored."""
    df = df.dropna(how='all').copy()
    out = pd.DataFrame(index=df.index)
    s = _col(df, m, 'gstin')
    out['gstin'] = s.map(clean_gstin) if s is not None else ''
    s = _col(df, m, 'party')
    out['party'] = s.map(clean_text) if s is not None else ''
    s = _col(df, m, 'invoice_no')
    out['inv_raw'] = s.map(clean_text) if s is not None else ''
    out['key'] = out['inv_raw'].map(inv_key)
    s = _col(df, m, 'date')
    out['date'] = s.map(parse_date) if s is not None else pd.NaT
    out['date'] = pd.to_datetime(out['date'])
    for k in ('taxable', 'igst', 'cgst', 'sgst'):
        s = _col(df, m, k)
        out[k] = s.map(clean_amount) if s is not None else 0.0
    out['tax'] = (out['igst'] + out['cgst'] + out['sgst']).round(2)
    out['pan'] = out['gstin'].str[2:12]
    keep = ~((out['gstin'] == '') & (out['inv_raw'] == ''))
    ignored = int((~keep).sum())
    out = out[keep].reset_index(drop=True)
    out['rid'] = range(len(out))
    return out, ignored


def normalize_cdnr(df, m):
    df = df.dropna(how='all').copy()
    out = pd.DataFrame(index=df.index)
    for src, dst in (('gstin', 'gstin'), ('party', 'party'), ('note_no', 'note_no')):
        s = _col(df, m, src)
        out[dst] = s.map(clean_gstin if src == 'gstin' else clean_text) if s is not None else ''
    s = _col(df, m, 'note_type')
    if s is not None:
        out['type'] = s.map(lambda v: 'Debit Note' if 'debit' in clean_text(v).lower() or clean_text(v).upper() == 'D'
                            else 'Credit Note')
    else:
        out['type'] = 'Credit Note'
    s = _col(df, m, 'date')
    out['date'] = pd.to_datetime(s.map(parse_date)) if s is not None else pd.NaT
    for k in ('taxable', 'igst', 'cgst', 'sgst'):
        s = _col(df, m, k)
        out[k] = s.map(clean_amount) if s is not None else 0.0
    out = out[~((out['gstin'] == '') & (out['note_no'] == ''))].reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------
# Reconciliation
# ----------------------------------------------------------------------------
def _gap(a, b):
    if pd.isna(a) or pd.isna(b):
        return 9999
    return abs((a - b).days)


def _fmt(d):
    return '' if pd.isna(d) else d.strftime('%d-%m-%Y')


def reconcile(books, gst, tol=2.0, max_gap=2, cutoff='2026-04-01', use_fuzzy=False, fuzzy_th=92):
    cutoff = pd.Timestamp(cutoff)
    B = {r['rid']: r for r in books.to_dict('records')}
    G = {r['rid']: r for r in gst.to_dict('records')}
    pairs = {}      # books rid -> (gst rid, match type)
    dupes = {}      # books rid -> kept books rid
    usedg = set()

    def close(a, c):
        return abs(a['taxable'] - c['taxable']) <= tol and abs(a['tax'] - c['tax']) <= tol

    def cost(a, c):
        g = _gap(a['date'], c['date'])
        return abs(a['taxable'] - c['taxable']) + abs(a['tax'] - c['tax']) + (0.001 * g if g < 9999 else 0.5)

    def pair(b, g, mt):
        pairs[b] = (g, mt)
        usedg.add(g)

    # ---- Stage 1: GSTIN + normalised invoice no ---------------------------------
    bgrp, ggrp = {}, {}
    for r in B.values():
        bgrp.setdefault((r['gstin'], r['key']), []).append(r['rid'])
    for r in G.values():
        ggrp.setdefault((r['gstin'], r['key']), []).append(r['rid'])
    for key, brids in bgrp.items():
        if key[0] == '' or key[1] == '':
            continue
        grids = ggrp.get(key)
        if grids:
            combos = sorted((cost(B[b], G[g]), b, g) for b in brids for g in grids)
            ub, ug = set(), set()
            for _, b, g in combos:
                if b in ub or g in ug:
                    continue
                mt = 'Exact' if B[b]['inv_raw'].upper() == G[g]['inv_raw'].upper() else 'Normalized'
                pair(b, g, mt)
                ub.add(b)
                ug.add(g)
            kept = [b for b in brids if b in ub]
        else:
            kept = brids[:1]
        for b in brids:
            if b in kept:
                continue
            ref = min(kept, key=lambda k: cost(B[b], B[k]))
            if close(B[b], B[ref]):
                dupes[b] = ref

    def free_books():
        return [r for r in B.values() if r['rid'] not in pairs and r['rid'] not in dupes]

    def free_gst():
        return [r for r in G.values() if r['rid'] not in usedg]

    # ---- Stage 0: Books GSTIN blank -> inv no + amounts ---------------------------
    gk = {}
    for r in free_gst():
        gk.setdefault(r['key'], []).append(r)
    for b in free_books():
        if b['gstin'] != '' or b['key'] == '':
            continue
        for g in gk.get(b['key'], []):
            if g['rid'] not in usedg and close(b, g):
                pair(b['rid'], g['rid'], 'Blank GSTIN')
                break

    # ---- Stage 2: same GSTIN, amounts close, date gap strict ---------------------
    ub_by, ug_by = {}, {}
    for r in free_books():
        if r['gstin'] != '':
            ub_by.setdefault(r['gstin'], []).append(r)
    for r in free_gst():
        ug_by.setdefault(r['gstin'], []).append(r)
    for gstin, brs in ub_by.items():
        cands = []
        for b in brs:
            if b['tax'] <= 0:
                continue
            for g in ug_by.get(gstin, []):
                if close(b, g):
                    gp = _gap(b['date'], g['date'])
                    if gp <= max_gap:
                        cands.append((gp, cost(b, g), b['rid'], g['rid']))
        for _, _, b, g in sorted(cands):
            if b in pairs or g in usedg:
                continue
            kb, kg = B[b]['key'], G[g]['key']
            near = len(kb) >= 3 and len(kg) >= 3 and (kb in kg or kg in kb)
            pair(b, g, 'Near Match' if near else 'Amount Fallback')

    # ---- Stage 3: same invoice no + amounts, GSTIN differs -------------------------
    gk = {}
    for r in free_gst():
        gk.setdefault(r['key'], []).append(r)
    cands = []
    for b in free_books():
        if b['key'] == '' or b['tax'] <= 0:
            continue
        for g in gk.get(b['key'], []):
            if g['gstin'] != b['gstin'] and close(b, g):
                gp = _gap(b['date'], g['date'])
                if gp <= max_gap:
                    cands.append((gp, cost(b, g), b['rid'], g['rid']))
    for _, _, b, g in sorted(cands):
        if b in pairs or g in usedg:
            continue
        pair(b, g, 'GSTIN Mismatch')

    # ---- Stage F (optional): fuzzy invoice no, same GSTIN ---------------------------
    if use_fuzzy:
        ub_by, ug_by = {}, {}
        for r in free_books():
            if r['gstin'] != '' and r['key'] != '':
                ub_by.setdefault(r['gstin'], []).append(r)
        for r in free_gst():
            ug_by.setdefault(r['gstin'], []).append(r)
        cands = []
        for gstin, brs in ub_by.items():
            for b in brs:
                for g in ug_by.get(gstin, []):
                    sc = SequenceMatcher(None, b['key'], g['key']).ratio() * 100
                    gp = _gap(b['date'], g['date'])
                    if sc >= fuzzy_th and gp <= max_gap:
                        cands.append((-sc, gp, b['rid'], g['rid']))
        for _, _, b, g in sorted(cands):
            if b in pairs or g in usedg:
                continue
            pair(b, g, 'Fuzzy (verify)')

    # ---- Classification -----------------------------------------------------------
    def classify(b, g, mt):
        gd = g['date']
        if not pd.isna(gd) and gd < cutoff:
            return 'Previous Invoice'
        if b['tax'] <= 0.005 and g['tax'] > 0.005:
            return 'ITC Not Claimed'
        if mt in ('Near Match', 'Amount Fallback', 'Fuzzy (verify)'):
            return 'Inv No Mismatch'
        if mt == 'GSTIN Mismatch':
            return 'GSTIN Mismatch'
        if abs(b['taxable'] - g['taxable']) > tol or any(abs(b[h] - g[h]) > tol for h in ('igst', 'cgst', 'sgst')):
            return 'Value Mismatch'
        if not pd.isna(b['date']) and not pd.isna(gd) and b['date'] != gd:
            return 'Date Mismatch'
        return 'Matched'

    def remark(st, b, g, mt):
        if st == 'Matched':
            return 'Books GSTIN blank - matched on inv no + amounts; update GSTIN in Books' if mt == 'Blank GSTIN' else ''
        if st == 'Date Mismatch':
            d = _gap(b['date'], g['date'])
            return (f"Only date differs: Books {_fmt(b['date'])} vs 2B {_fmt(g['date'])} "
                    f"({d} day{'s' if d != 1 else ''}); inv no, taxable and GST match")
        if st == 'Value Mismatch':
            parts = []
            if abs(b['taxable'] - g['taxable']) > tol:
                parts.append(f"Taxable differs (Books {b['taxable']:,.2f} vs 2B {g['taxable']:,.2f})")
            for h, n in (('igst', 'IGST'), ('cgst', 'CGST'), ('sgst', 'SGST')):
                if abs(b[h] - g[h]) > tol:
                    parts.append(f"{n} missing in Books (2B {g[h]:,.2f})" if b[h] == 0
                                 else f"{n} not in 2B (Books {b[h]:,.2f})" if g[h] == 0
                                 else f"{n} differs (Books {b[h]:,.2f} vs 2B {g[h]:,.2f})")
            if b['date'] != g['date'] and not pd.isna(b['date']) and not pd.isna(g['date']):
                parts.append(f"date also differs (Books {_fmt(b['date'])} vs 2B {_fmt(g['date'])})")
            return 'Value Mismatch: ' + '; '.join(parts)
        if st == 'Inv No Mismatch':
            s = (f"Probable match ({mt}): inv no differs (Books {b['inv_raw']} vs 2B {g['inv_raw']}); "
                 f"taxable & GST same within Rs {tol:g}" if mt != 'Fuzzy (verify)' else
                 f"Fuzzy inv no match - verify (Books {b['inv_raw']} vs 2B {g['inv_raw']}); "
                 f"taxable diff {b['taxable'] - g['taxable']:,.2f}, GST diff {b['tax'] - g['tax']:,.2f}")
            if b['date'] != g['date'] and not pd.isna(b['date']) and not pd.isna(g['date']):
                s += f"; date also differs (Books {_fmt(b['date'])} vs 2B {_fmt(g['date'])})"
            return s
        if st == 'GSTIN Mismatch':
            return f"GSTIN differs: Books {b['gstin']} vs 2B {g['gstin']} (inv no and amounts match)"
        if st == 'ITC Not Claimed':
            return (f"No GST in Books (Books amount {b['taxable']:,.2f}) but 2B shows GST {g['tax']:,.2f} "
                    f"- ITC not claimed")
        if st == 'Previous Invoice':
            if b is None:
                return f"Invoice dated before {cutoff:%d-%m-%Y} - not found in Books"
            return (f"Invoice dated before {cutoff:%d-%m-%Y}; also in Books (voucher {b['inv_raw']}, "
                    f"Books date {_fmt(b['date'])})")
        return ''

    def pair_row(b, g, st, mt, rem):
        bz = {k: 0.0 for k in ('taxable', 'igst', 'cgst', 'sgst', 'tax')}
        bb = b if b is not None else bz
        gg = g if g is not None else bz
        return {
            'Invoice Date (Books)': b['date'] if b is not None else pd.NaT,
            'Invoice Date (2B)': g['date'] if g is not None else pd.NaT,
            'Status': st, 'Match Type': mt,
            'GSTIN (Books)': b['gstin'] if b is not None else '',
            'GSTIN (2B)': g['gstin'] if g is not None else '',
            'Party Name': (b['party'] if b is not None and b['party'] else (g['party'] if g is not None else '')),
            'Invoice No (Books)': b['inv_raw'] if b is not None else '',
            'Invoice No (2B)': g['inv_raw'] if g is not None else '',
            'Taxable (Books)': bb['taxable'], 'Taxable (2B)': gg['taxable'],
            'Taxable Diff (Books-2B)': round(bb['taxable'] - gg['taxable'], 2),
            'IGST (Books)': bb['igst'], 'IGST (2B)': gg['igst'], 'IGST Diff (Books-2B)': round(bb['igst'] - gg['igst'], 2),
            'CGST (Books)': bb['cgst'], 'CGST (2B)': gg['cgst'], 'CGST Diff (Books-2B)': round(bb['cgst'] - gg['cgst'], 2),
            'SGST (Books)': bb['sgst'], 'SGST (2B)': gg['sgst'], 'SGST Diff (Books-2B)': round(bb['sgst'] - gg['sgst'], 2),
            'Total GST (Books)': round(bb['tax'], 2), 'Total GST (2B)': round(gg['tax'], 2),
            'GST Diff (Books-2B)': round(bb['tax'] - gg['tax'], 2),
            'Remark': rem,
        }

    buckets = {'main': [], 'Previous Invoice': [], 'ITC Not Claimed': []}
    for b, (g, mt) in pairs.items():
        bb, gg = B[b], G[g]
        st = classify(bb, gg, mt)
        row = pair_row(bb, gg, st, mt, remark(st, bb, gg, mt))
        buckets['main' if st not in ('Previous Invoice', 'ITC Not Claimed') else st].append(row)
    for d, k in dupes.items():
        bb = B[d]
        buckets['main'].append(pair_row(bb, None, 'Duplicate Invoice', f"Duplicate of {B[k]['inv_raw']}",
                                        'Duplicate entry in Books (same GSTIN + inv no + amounts already matched) '
                                        '- ITC double-count risk'))
    # 2B-only rows
    missing_tally = []
    for g in free_gst():
        if not pd.isna(g['date']) and g['date'] < cutoff:
            buckets['Previous Invoice'].append(pair_row(None, g, 'Previous Invoice', '', remark('Previous Invoice', None, g, '')))
        else:
            missing_tally.append({'Invoice Date': g['date'], 'GSTIN': g['gstin'], 'Party Name': g['party'],
                                  'Invoice No': g['inv_raw'], 'Taxable': g['taxable'], 'IGST': g['igst'],
                                  'CGST': g['cgst'], 'SGST': g['sgst'], 'Total GST': round(g['tax'], 2),
                                  'Remark': 'In 2B, not booked in Books - check / book purchase'})
    paired_keys = {(B[b]['gstin'], B[b]['key']) for b in pairs}
    missing_2b = []
    for b in free_books():
        if (b['gstin'], b['key']) in paired_keys and b['gstin'] != '':
            rem = 'Same invoice no already matched with a different amount - check'
        elif b['gstin'] == '' and b['tax'] <= 0:
            rem = 'No GSTIN / no GST in Books (unregistered purchase) - ITC not applicable'
        elif b['tax'] <= 0:
            rem = 'No GST in Books and invoice not in 2B'
        else:
            rem = 'Not in GSTR-2B - follow up with supplier (GSTR-1 not filed / late)'
        missing_2b.append({'Invoice Date': b['date'], 'GSTIN': b['gstin'], 'Party Name': b['party'],
                           'Invoice No': b['inv_raw'], 'Taxable': b['taxable'], 'IGST': b['igst'],
                           'CGST': b['cgst'], 'SGST': b['sgst'], 'Total GST': round(b['tax'], 2), 'Remark': rem})

    def frame(rows, cols, sortcols):
        df = pd.DataFrame(rows, columns=cols)
        return df.sort_values(sortcols, kind='stable').reset_index(drop=True) if len(df) else df

    main = pd.DataFrame(buckets['main'], columns=PAIR_COLS)
    if len(main):
        main['_o'] = main['Status'].map({s: i for i, s in enumerate(STATUS_ORDER)})
        main = main.sort_values(['_o', 'Invoice Date (Books)'], kind='stable').drop(columns='_o').reset_index(drop=True)
    res = {
        'matched': main,
        'not_claimed': frame(buckets['ITC Not Claimed'], PAIR_COLS, ['Invoice Date (Books)']),
        'previous': frame(buckets['Previous Invoice'], PAIR_COLS, ['Invoice Date (2B)']),
        'missing_2b': frame(missing_2b, MISS_COLS, ['Invoice Date', 'Invoice No']),
        'missing_tally': frame(missing_tally, MISS_COLS, ['Invoice Date', 'Invoice No']),
        'control_2b': {'n': len(G), 'igst': round(sum(r['igst'] for r in G.values()), 2),
                       'cgst': round(sum(r['cgst'] for r in G.values()), 2),
                       'sgst': round(sum(r['sgst'] for r in G.values()), 2)},
        'control_books': {'n': len(B), 'igst': round(sum(r['igst'] for r in B.values()), 2),
                          'cgst': round(sum(r['cgst'] for r in B.values()), 2),
                          'sgst': round(sum(r['sgst'] for r in B.values()), 2)},
        'cutoff': cutoff, 'tol': tol, 'max_gap': max_gap,
    }
    res['stats'] = compute_stats(res)
    return res


def _agg(df, status=None):
    d = df if status is None else df[df['Status'] == status]
    return {'n': len(d),
            'tx': round(d['Taxable (2B)'].sum(), 2), 'ig': round(d['IGST (2B)'].sum(), 2),
            'cg': round(d['CGST (2B)'].sum(), 2), 'sg': round(d['SGST (2B)'].sum(), 2),
            'bk': round(d['Total GST (Books)'].sum(), 2),
            'bk_ig': round(d['IGST (Books)'].sum(), 2), 'bk_cg': round(d['CGST (Books)'].sum(), 2),
            'bk_sg': round(d['SGST (Books)'].sum(), 2)}


def compute_stats(res):
    m = res['matched']
    mt = res['missing_tally']
    s = {k: _agg(m, k) for k in STATUS_ORDER}
    s['ITC Not Claimed'] = _agg(res['not_claimed'])
    s['Previous Invoice'] = _agg(res['previous'])
    s['Missing in Tally'] = {'n': len(mt), 'tx': round(mt['Taxable'].sum(), 2), 'ig': round(mt['IGST'].sum(), 2),
                             'cg': round(mt['CGST'].sum(), 2), 'sg': round(mt['SGST'].sum(), 2), 'bk': 0.0,
                             'bk_ig': 0.0, 'bk_cg': 0.0, 'bk_sg': 0.0}
    m2 = res['missing_2b']
    s['Missing in 2B'] = {'n': len(m2), 'bk_ig': round(m2['IGST'].sum(), 2), 'bk_cg': round(m2['CGST'].sum(), 2),
                          'bk_sg': round(m2['SGST'].sum(), 2)}
    tot = lambda keys, f: round(sum(s[k][f] for k in keys), 2)
    matched_keys = ['Matched', 'Date Mismatch']
    remain_keys = ['Value Mismatch', 'Inv No Mismatch', 'GSTIN Mismatch', 'ITC Not Claimed',
                   'Previous Invoice', 'Missing in Tally']
    s['matched_itc'] = round(sum(s[k][f] for k in matched_keys for f in ('ig', 'cg', 'sg')), 2)
    s['remaining_itc'] = round(sum(s[k][f] for k in remain_keys for f in ('ig', 'cg', 'sg')), 2)
    s['total_2b_itc'] = round(s['matched_itc'] + s['remaining_itc'], 2)
    s['matched_keys'], s['remain_keys'] = matched_keys, remain_keys
    return s


# ----------------------------------------------------------------------------
# Excel output
# ----------------------------------------------------------------------------
def build_workbook(res, cdnr=None, settings=None):
    """res = reconcile() output, cdnr = normalize_cdnr() frame or None. Returns BytesIO."""
    import xlsxwriter
    from xlsxwriter.utility import xl_col_to_name as cn

    settings = settings or {}
    out = BytesIO()
    wb = xlsxwriter.Workbook(out, {'nan_inf_to_errors': True, 'default_date_format': 'dd-mm-yyyy'})
    cache = {}

    def F(**kw):
        key = tuple(sorted(kw.items()))
        if key not in cache:
            base = {'font_name': 'Arial', 'font_size': 10}
            base.update(kw)
            cache[key] = wb.add_format(base)
        return cache[key]

    NUM = '#,##0.00;[Red]-#,##0.00;"-"'
    HDR = F(bold=True, font_color='white', bg_color='#1E3A5F', border=1, align='center', valign='vcenter', text_wrap=True)
    BG = {'Value Mismatch': '#FFEB9C', 'GSTIN Mismatch': '#FFC7CE', 'Duplicate Invoice': '#FCE4D6',
          'Inv No Mismatch': '#DDEBF7', 'Date Mismatch': '#E2EFDA'}

    def cell_fmt(bg=None, kind='text'):
        kw = {'border': 1}
        if bg:
            kw['bg_color'] = bg
        if kind == 'num':
            kw['num_format'] = NUM
        elif kind == 'date':
            kw['num_format'] = 'dd-mm-yyyy'
        return F(**kw)

    def put(ws, r, c, v, fmt):
        if isinstance(v, pd.Timestamp) or v is pd.NaT:
            if pd.isna(v):
                ws.write_blank(r, c, None, fmt)
            else:
                ws.write_datetime(r, c, v.to_pydatetime(), fmt)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            ws.write_number(r, c, float(v), fmt)
        elif v is None or v == '':
            ws.write_blank(r, c, None, fmt)
        else:
            ws.write_string(r, c, str(v), fmt)

    # ---- pair sheets ---------------------------------------------------------------
    PW = [12, 12, 17, 18, 18, 18, 34, 20, 20, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 80]
    FORM = {  # col index -> (formula template using letters, python value column)
        11: ('={J}-{K}', 'Taxable Diff (Books-2B)'), 14: ('={M}-{N}', 'IGST Diff (Books-2B)'),
        17: ('={P}-{Q}', 'CGST Diff (Books-2B)'), 20: ('={S}-{T}', 'SGST Diff (Books-2B)'),
        21: ('={M}+{P}+{S}', 'Total GST (Books)'), 22: ('={N}+{Q}+{T}', 'Total GST (2B)'),
        23: ('={V}-{W}', 'GST Diff (Books-2B)')}
    DATE_COLS, NUM_COLS = (0, 1), tuple(range(9, 24))

    def pair_sheet(name, df):
        ws = wb.add_worksheet(name)
        ws.set_row(0, 32)
        for j, h in enumerate(PAIR_COLS):
            ws.write(0, j, h, HDR)
            ws.set_column(j, j, PW[j])
        letters = {x: cn(i) for i, x in zip(range(26), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
        for i, row in enumerate(df.itertuples(index=False), start=1):
            bg = BG.get(row[2])
            for j, v in enumerate(row):
                if j in FORM:
                    tpl, pc = FORM[j]
                    f = tpl.format(**{k: f'{k}{i + 1}' for k in 'JKMNPQSTVW'})
                    ws.write_formula(i, j, f, cell_fmt(bg, 'num'), float(v))
                elif j in DATE_COLS:
                    put(ws, i, j, v, cell_fmt(bg, 'date'))
                elif j in NUM_COLS:
                    put(ws, i, j, v, cell_fmt(bg, 'num'))
                else:
                    put(ws, i, j, v, cell_fmt(bg))
        ws.autofilter(0, 0, max(len(df), 1), len(PAIR_COLS) - 1)
        ws.freeze_panes(1, 3)
        return ws

    def miss_sheet(name, df, bg):
        ws = wb.add_worksheet(name)
        ws.set_row(0, 30)
        for j, (h, w) in enumerate(zip(MISS_COLS, [12, 18, 40, 22, 14, 13, 13, 13, 14, 80])):
            ws.write(0, j, h, HDR)
            ws.set_column(j, j, w)
        for i, row in enumerate(df.itertuples(index=False), start=1):
            for j, v in enumerate(row):
                if j == 8:
                    ws.write_formula(i, j, f'=F{i + 1}+G{i + 1}+H{i + 1}', cell_fmt(bg, 'num'), float(v))
                elif j == 0:
                    put(ws, i, j, v, cell_fmt(bg, 'date'))
                elif 4 <= j <= 7:
                    put(ws, i, j, v, cell_fmt(bg, 'num'))
                else:
                    put(ws, i, j, v, cell_fmt(bg))
        ws.autofilter(0, 0, max(len(df), 1), len(MISS_COLS) - 1)
        ws.freeze_panes(1, 0)
        return ws

    # Sheet order: Result first (created now, filled later), then data sheets
    ws_res = wb.add_worksheet('Result')
    pair_sheet('Matched Invoices', res['matched'])
    miss_sheet('Missing in 2B', res['missing_2b'], '#FFC7CE')
    miss_sheet('Missing in Tally', res['missing_tally'], None)
    pair_sheet('Previous Invoice', res['previous'])
    pair_sheet('ITC Not Claimed', res['not_claimed'])
    if cdnr is not None and len(cdnr):
        ws = wb.add_worksheet('CDNR')
        for j, (h, w) in enumerate(zip(CDNR_COLS, [12, 18, 40, 20, 13, 14, 13, 13, 13, 14, 40])):
            ws.write(0, j, h, HDR)
            ws.set_column(j, j, w)
        ws.set_row(0, 30)
        cd = cdnr.sort_values(['date', 'gstin'], kind='stable').reset_index(drop=True)
        for i, r in enumerate(cd.itertuples(index=False), start=1):
            put(ws, i, 0, r.date, cell_fmt(None, 'date'))
            put(ws, i, 1, r.gstin, cell_fmt())
            put(ws, i, 2, r.party, cell_fmt())
            put(ws, i, 3, r.note_no, cell_fmt())
            put(ws, i, 4, r.type, cell_fmt())
            put(ws, i, 5, r.taxable, cell_fmt(None, 'num'))
            put(ws, i, 6, r.igst, cell_fmt(None, 'num'))
            put(ws, i, 7, r.cgst, cell_fmt(None, 'num'))
            put(ws, i, 8, r.sgst, cell_fmt(None, 'num'))
            ws.write_formula(i, 9, f'=G{i + 1}+H{i + 1}+I{i + 1}', cell_fmt(None, 'num'), round(r.igst + r.cgst + r.sgst, 2))
            put(ws, i, 10, 'ITC to be reduced' if r.type == 'Credit Note' else 'ITC increase', cell_fmt())
        ws.autofilter(0, 0, len(cd), len(CDNR_COLS) - 1)
        ws.freeze_panes(1, 0)

    # ---- Result sheet ----------------------------------------------------------------
    ws = ws_res
    S = res['stats']
    for col, w in zip('ABCDEFGHIJK', [48, 20, 12, 17, 16, 15, 15, 17, 11, 18, 17]):
        ws.set_column(f'{col}:{col}', w)
    ws.hide_gridlines(2)
    title = F(bold=True, font_size=14)
    note = F(italic=True, font_size=9, font_color='#555555')
    sec = F(bold=True, font_color='white', bg_color='#2F5597')
    lab = F(border=1)
    labb = F(border=1, bold=True)
    n_ = lambda bold=False, bg=None: F(border=1, num_format=NUM, **({'bold': True} if bold else {}), **({'bg_color': bg} if bg else {}))
    i_ = lambda bold=False, bg=None: F(border=1, num_format='#,##0', **({'bold': True} if bold else {}), **({'bg_color': bg} if bg else {}))
    p_ = lambda bold=False, bg=None: F(border=1, num_format='0.0%', **({'bold': True} if bold else {}), **({'bg_color': bg} if bg else {}))
    inp = F(border=1, num_format=NUM, font_color='#0000FF')
    inpi = F(border=1, num_format='#,##0', font_color='#0000FF')

    ws.write('A1', 'ITC RECONCILIATION RESULT - GSTR-2B vs BOOKS (PURCHASE REGISTER)', title)
    ws.write('A2', f"Rounding tolerance: Rs {res['tol']:g} | Date gap for fallback matching: {res['max_gap']} day(s) | "
                   f"Previous Invoice cutoff: {res['cutoff']:%d-%m-%Y} | All figures in Rs", note)

    def secrow(r, text):
        for c in range(11):
            ws.write_blank(r - 1, c, None, sec)
        ws.write(r - 1, 0, text, sec)

    def hdr(r, d):
        ws.set_row(r - 1, 30)
        for c, t in d.items():
            ws.write(r - 1, c, t, HDR)

    MS = "'Matched Invoices'!"
    pc = {'status': '$C:$C', 'tx2b': '$K:$K', 'ig2b': '$N:$N', 'cg2b': '$Q:$Q', 'sg2b': '$T:$T', 'bk': '$V:$V',
          'bkig': '$M:$M', 'bkcg': '$P:$P', 'bksg': '$S:$S', 'bkinv': '$H:$H'}
    secrow(4, 'A. ITC AS PER GSTR-2B - CATEGORY WISE')
    hdr(5, {0: 'Category', 1: 'Sheet / Status', 2: 'No. of Invoices', 3: '2B Taxable', 4: '2B IGST', 5: '2B CGST',
            6: '2B SGST', 7: '2B Total ITC', 8: '% of 2B ITC', 9: 'Books ITC (same invoices)', 10: 'Diff (Books - 2B)'})

    def catrow(r, label, sheet, status, s, mode):
        q = f"'{sheet}'!"
        ws.write(r - 1, 0, label, lab)
        ws.write(r - 1, 1, status or sheet, lab)
        if mode == 'status':
            f = {'n': f'=COUNTIF({q}{pc["status"]},"{status}")'}
            for k, c in (('tx', 'tx2b'), ('ig', 'ig2b'), ('cg', 'cg2b'), ('sg', 'sg2b'), ('bk', 'bk')):
                f[k] = f'=SUMIFS({q}{pc[c]},{q}{pc["status"]},"{status}")'
        elif mode == 'sheet':
            f = {'n': f'=COUNTA({q}{pc["status"]})-1'}
            for k, c in (('tx', 'tx2b'), ('ig', 'ig2b'), ('cg', 'cg2b'), ('sg', 'sg2b'), ('bk', 'bk')):
                f[k] = f'=SUM({q}{c and pc[c]})'
        else:  # missing in tally sheet (different layout)
            f = {'n': f'=COUNTA({q}$J:$J)-1', 'tx': f'=SUM({q}$E:$E)', 'ig': f'=SUM({q}$F:$F)',
                 'cg': f'=SUM({q}$G:$G)', 'sg': f'=SUM({q}$H:$H)', 'bk': '=0'}
        ws.write_formula(r - 1, 2, f['n'], i_(), s['n'])
        ws.write_formula(r - 1, 3, f['tx'], n_(), s['tx'])
        ws.write_formula(r - 1, 4, f['ig'], n_(), s['ig'])
        ws.write_formula(r - 1, 5, f['cg'], n_(), s['cg'])
        ws.write_formula(r - 1, 6, f['sg'], n_(), s['sg'])
        tot = round(s['ig'] + s['cg'] + s['sg'], 2)
        ws.write_formula(r - 1, 7, f'=E{r}+F{r}+G{r}', n_(), tot)
        ws.write_formula(r - 1, 8, f'=IF($H$16=0,0,H{r}/$H$16)', p_(), tot / S['total_2b_itc'] if S['total_2b_itc'] else 0)
        ws.write_formula(r - 1, 9, f['bk'], n_(), s['bk'])
        ws.write_formula(r - 1, 10, f'=J{r}-H{r}', n_(), round(s['bk'] - tot, 2))

    catrow(6, 'ITC Matched - all fields match', 'Matched Invoices', 'Matched', S['Matched'], 'status')
    catrow(7, 'Date Mismatch (only date differs)', 'Matched Invoices', 'Date Mismatch', S['Date Mismatch'], 'status')
    catrow(9, 'Value Mismatch (taxable / head-wise GST differs)', 'Matched Invoices', 'Value Mismatch', S['Value Mismatch'], 'status')
    catrow(10, 'Inv No Mismatch (probable match - verify)', 'Matched Invoices', 'Inv No Mismatch', S['Inv No Mismatch'], 'status')
    catrow(11, 'GSTIN Mismatch (inv no + amount same)', 'Matched Invoices', 'GSTIN Mismatch', S['GSTIN Mismatch'], 'status')
    catrow(12, 'ITC Not Claimed (no GST in Books, GST in 2B)', 'ITC Not Claimed', None, S['ITC Not Claimed'], 'sheet')
    catrow(13, f"Previous Invoice (dated before {res['cutoff']:%d-%m-%Y})", 'Previous Invoice', None, S['Previous Invoice'], 'sheet')
    catrow(14, 'Missing in Tally (in 2B, not in Books)', 'Missing in Tally', None, S['Missing in Tally'], 'tally')

    def sumrow(r, label, parts, keys, bg, bold_fill):
        ws.write(r - 1, 0, label, labb)
        ws.write_blank(r - 1, 1, None, labb)
        agg = {k: sum(S[x][k] for x in keys) for k in ('n', 'tx', 'ig', 'cg', 'sg', 'bk')}
        for c, k, fmt in ((2, 'n', i_(True, bg)), (3, 'tx', n_(True, bg)), (4, 'ig', n_(True, bg)),
                          (5, 'cg', n_(True, bg)), (6, 'sg', n_(True, bg)), (9, 'bk', n_(True, bg))):
            col = cn(c)
            ws.write_formula(r - 1, c, '=' + '+'.join(f'{col}{p}' for p in parts) if not isinstance(parts, str) else parts.format(c=col),
                             fmt, round(agg[k], 2))
        tot = round(agg['ig'] + agg['cg'] + agg['sg'], 2)
        ws.write_formula(r - 1, 7, f'=E{r}+F{r}+G{r}', n_(True, bg), tot)
        ws.write_formula(r - 1, 8, f'=IF($H$16=0,0,H{r}/$H$16)', p_(True, bg), tot / S['total_2b_itc'] if S['total_2b_itc'] else 0)
        ws.write_formula(r - 1, 10, f'=J{r}-H{r}', n_(True, bg), round(agg['bk'] - tot, 2))

    sumrow(8, 'ITC MATCHED (Matched + Date Mismatch)', [6, 7], ['Matched', 'Date Mismatch'], '#D9E1F2', True)
    sumrow(15, 'ITC REMAINING (not matched)', [9, 10, 11, 12, 13, 14],
           ['Value Mismatch', 'Inv No Mismatch', 'GSTIN Mismatch', 'ITC Not Claimed', 'Previous Invoice', 'Missing in Tally'],
           '#D9E1F2', True)
    # total row
    allk = S['matched_keys'] + S['remain_keys']
    agg = {k: sum(S[x][k] for x in allk) for k in ('n', 'tx', 'ig', 'cg', 'sg', 'bk')}
    ws.write(15, 0, 'TOTAL ITC AS PER 2B (Matched + Remaining)', labb)
    ws.write_blank(15, 1, None, labb)
    for c, k, fm in ((2, 'n', i_(True, '#FFF2CC')), (3, 'tx', n_(True, '#FFF2CC')), (4, 'ig', n_(True, '#FFF2CC')),
                     (5, 'cg', n_(True, '#FFF2CC')), (6, 'sg', n_(True, '#FFF2CC')), (9, 'bk', n_(True, '#FFF2CC'))):
        col = cn(c)
        ws.write_formula(15, c, f'={col}8+{col}15', fm, round(agg[k], 2))
    tot2b = round(agg['ig'] + agg['cg'] + agg['sg'], 2)
    ws.write_formula(15, 7, '=E16+F16+G16', n_(True, '#FFF2CC'), tot2b)
    ws.write_formula(15, 8, '=IF($H$16=0,0,H16/$H$16)', p_(True, '#FFF2CC'), 1.0 if tot2b else 0)
    ws.write_formula(15, 10, '=J16-H16', n_(True, '#FFF2CC'), round(agg['bk'] - tot2b, 2))
    # control
    c2 = res['control_2b']
    ws.write('A17', 'Control: 2B total per uploaded file (input)', lab)
    ws.write_blank('B17', None, lab)
    ws.write_number('C17', c2['n'], inpi)
    for col in 'DHIJK':
        ws.write_blank(f'{col}17', None, lab)
    ws.write_blank('D17', None, lab)
    ws.write_number('E17', c2['igst'], inp)
    ws.write_number('F17', c2['cgst'], inp)
    ws.write_number('G17', c2['sgst'], inp)
    ws.write_formula('H17', '=E17+F17+G17', n_(), round(c2['igst'] + c2['cgst'] + c2['sgst'], 2))
    ws.write('A18', 'Check: Total ITC as per 2B - Control (must be 0)', F(border=1, italic=True))
    ws.write_blank('B18', None, lab)
    for col, v in (('C', agg['n'] - c2['n']), ('E', agg['ig'] - c2['igst']), ('F', agg['cg'] - c2['cgst']),
                   ('G', agg['sg'] - c2['sgst']), ('H', tot2b - (c2['igst'] + c2['cgst'] + c2['sgst']))):
        ws.write_formula(f'{col}18', f'={col}16-{col}17', n_() if col != 'C' else i_(), round(v, 2))
    for col in 'DIJK':
        ws.write_blank(f'{col}18', None, lab)
    ws.write('A19', 'Blue = control figure taken from the uploaded 2B file; every other figure is a formula on the data sheets.', note)

    # ---- B
    secrow(21, 'B. KITNA ITC MATCH HUA / KITNA RAH GAYA (as per 2B)')
    hdr(22, {0: 'Particulars', 4: 'IGST', 5: 'CGST', 6: 'SGST', 7: 'Total ITC', 8: '%'})
    for r, lab_, src, bg in ((23, 'Total ITC available as per 2B', 16, None), (24, 'ITC matched with Books (done)', 8, '#E2EFDA'),
                             (25, 'ITC remaining (to be resolved)', 15, '#FCE4D6')):
        ws.write(r - 1, 0, lab_, labb)
        for c in (1, 2, 3, 9, 10):
            ws.write_blank(r - 1, c, None, labb)
        if src == 16:
            vals = (agg['ig'], agg['cg'], agg['sg'])
        elif src == 8:
            vals = tuple(sum(S[k][x] for k in S['matched_keys']) for x in ('ig', 'cg', 'sg'))
        else:
            vals = tuple(sum(S[k][x] for k in S['remain_keys']) for x in ('ig', 'cg', 'sg'))
        for c, v in zip((4, 5, 6), vals):
            ws.write_formula(r - 1, c, f'={cn(c)}{src}', n_(True, bg), round(v, 2))
        ws.write_formula(r - 1, 7, f'=E{r}+F{r}+G{r}', n_(True, bg), round(sum(vals), 2))
        ws.write_formula(r - 1, 8, f'=IF($H$23=0,0,H{r}/$H$23)', p_(True, bg), (sum(vals) / tot2b) if tot2b else 0)
    ws.write('A26', 'Remaining = Value Mismatch + Inv No Mismatch + GSTIN Mismatch + ITC Not Claimed + Previous Invoice + Missing in Tally (section A rows 9-14).', note)

    # ---- C
    secrow(28, 'C. BOOKS (PURCHASE REGISTER) SIDE')
    hdr(29, {0: 'Particulars', 2: 'Vouchers', 4: 'IGST', 5: 'CGST', 6: 'SGST', 7: 'Total ITC'})
    NC, PV, M2 = "'ITC Not Claimed'!", "'Previous Invoice'!", "'Missing in 2B'!"
    paired_vals = [S['Matched'], S['Date Mismatch'], S['Value Mismatch'], S['Inv No Mismatch'], S['GSTIN Mismatch'],
                   S['ITC Not Claimed'], S['Previous Invoice']]
    pv = {x: round(sum(s[x] for s in paired_vals), 2) for x in ('bk_ig', 'bk_cg', 'bk_sg')}
    pn = sum(s['n'] for s in paired_vals[:5]) + S['ITC Not Claimed']['n'] + int((res['previous']['Invoice No (Books)'] != '').sum())
    dv = S['Duplicate Invoice']
    mv = S['Missing in 2B']
    rows = [
        (30, 'Books ITC on invoices found in 2B (all categories)',
         f'=COUNTA({MS}{pc["status"]})-1-COUNTIF({MS}{pc["status"]},"Duplicate Invoice")+(COUNTA({NC}{pc["status"]})-1)+(COUNTIF({PV}{pc["bkinv"]},"?*")-1)', pn,
         {x: f'=SUMIFS({MS}{pc[y]},{MS}{pc["status"]},"<>Duplicate Invoice")+SUM({NC}{pc[y]})+SUM({PV}{pc[y]})'
          for x, y in (('bk_ig', 'bkig'), ('bk_cg', 'bkcg'), ('bk_sg', 'bksg'))}, pv),
        (31, 'Books ITC booked but Missing in 2B', f'=COUNTA({M2}$J:$J)-1', mv['n'],
         {'bk_ig': f'=SUM({M2}$F:$F)', 'bk_cg': f'=SUM({M2}$G:$G)', 'bk_sg': f'=SUM({M2}$H:$H)'}, mv),
        (32, 'Duplicate entries in Books (rows in Matched Invoices sheet)', f'=COUNTIF({MS}{pc["status"]},"Duplicate Invoice")', dv['n'],
         {x: f'=SUMIFS({MS}{pc[y]},{MS}{pc["status"]},"Duplicate Invoice")' for x, y in (('bk_ig', 'bkig'), ('bk_cg', 'bkcg'), ('bk_sg', 'bksg'))}, dv),
    ]
    for r, label, fn, vn, fs, vs in rows:
        ws.write(r - 1, 0, label, lab)
        for c in (1, 3, 8, 9, 10):
            ws.write_blank(r - 1, c, None, lab)
        ws.write_formula(r - 1, 2, fn, i_(), vn)
        for c, k in ((4, 'bk_ig'), (5, 'bk_cg'), (6, 'bk_sg')):
            ws.write_formula(r - 1, c, fs[k], n_(), vs[k])
        ws.write_formula(r - 1, 7, f'=E{r}+F{r}+G{r}', n_(), round(vs['bk_ig'] + vs['bk_cg'] + vs['bk_sg'], 2))
    tb = {k: round(pv[k] + mv[k] + dv[k], 2) for k in ('bk_ig', 'bk_cg', 'bk_sg')}
    ws.write('A33', 'Total ITC booked in Books', labb)
    for c in (1, 3, 8, 9, 10):
        ws.write_blank(32, c, None, labb)
    ws.write_formula('C33', '=C30+C31+C32', i_(True, '#D9E1F2'), pn + mv['n'] + dv['n'])
    for c, k in ((4, 'bk_ig'), (5, 'bk_cg'), (6, 'bk_sg')):
        ws.write_formula(32, c, f'={cn(c)}30+{cn(c)}31+{cn(c)}32', n_(True, '#D9E1F2'), tb[k])
    ws.write_formula('H33', '=E33+F33+G33', n_(True, '#D9E1F2'), round(sum(tb.values()), 2))
    cb = res['control_books']
    ws.write('A34', 'Control: Books total per uploaded file (input)', lab)
    for c in (1, 3, 8, 9, 10):
        ws.write_blank(33, c, None, lab)
    ws.write_number('C34', cb['n'], inpi)
    ws.write_number('E34', cb['igst'], inp)
    ws.write_number('F34', cb['cgst'], inp)
    ws.write_number('G34', cb['sgst'], inp)
    ws.write_formula('H34', '=E34+F34+G34', n_(), round(cb['igst'] + cb['cgst'] + cb['sgst'], 2))
    ws.write('A35', 'Check: Books ITC - Control (must be 0)', F(border=1, italic=True))
    for c in (1, 3, 8, 9, 10):
        ws.write_blank(34, c, None, lab)
    for col, v in (('C', pn + mv['n'] + dv['n'] - cb['n']), ('E', tb['bk_ig'] - cb['igst']), ('F', tb['bk_cg'] - cb['cgst']),
                   ('G', tb['bk_sg'] - cb['sgst']), ('H', sum(tb.values()) - (cb['igst'] + cb['cgst'] + cb['sgst']))):
        ws.write_formula(f'{col}35', f'={col}33-{col}34', n_() if col != 'C' else i_(), round(v, 2))
    ws.write('A36', 'Books ITC = IGST + CGST + SGST of each voucher. Rows Missing in 2B are not claimable until they appear in GSTR-2B.', note)

    # ---- D
    secrow(38, 'D. CREDIT / DEBIT NOTES IN GSTR-2B (CDNR) - not part of the matching above')
    hdr(39, {0: 'Particulars', 2: 'Notes', 4: 'IGST', 5: 'CGST', 6: 'SGST', 7: 'Total'})
    if cdnr is not None and len(cdnr):
        vals = {}
        for t in ('Credit Note', 'Debit Note'):
            d = cdnr[cdnr['type'] == t]
            vals[t] = (len(d), d['igst'].sum(), d['cgst'].sum(), d['sgst'].sum())
        for r, t, lbl in ((40, 'Credit Note', 'Credit Notes (ITC to be reduced)'), (41, 'Debit Note', 'Debit Notes (ITC increase)')):
            ws.write(r - 1, 0, lbl, lab)
            for c in (1, 3, 8, 9, 10):
                ws.write_blank(r - 1, c, None, lab)
            n, a, b_, c_ = vals[t]
            ws.write_formula(r - 1, 2, f'=COUNTIF(CDNR!$E:$E,"{t}")', i_(), n)
            for c, src, v in ((4, 'G', a), (5, 'H', b_), (6, 'I', c_)):
                ws.write_formula(r - 1, c, f'=SUMIF(CDNR!$E:$E,"{t}",CDNR!${src}:${src})', n_(), round(v, 2))
            ws.write_formula(r - 1, 7, f'=E{r}+F{r}+G{r}', n_(), round(a + b_ + c_, 2))
        ws.write('A42', 'Net ITC effect (Debit - Credit)', labb)
        for c in (1, 3, 8, 9, 10):
            ws.write_blank(41, c, None, labb)
        ws.write_formula('C42', '=C40+C41', i_(True, '#D9E1F2'), vals['Credit Note'][0] + vals['Debit Note'][0])
        for c, i in ((4, 1), (5, 2), (6, 3)):
            ws.write_formula(41, c, f'={cn(c)}41-{cn(c)}40', n_(True, '#D9E1F2'), round(vals['Debit Note'][i] - vals['Credit Note'][i], 2))
        net = sum(vals['Debit Note'][1:]) - sum(vals['Credit Note'][1:])
        ws.write_formula('H42', '=E42+F42+G42', n_(True, '#D9E1F2'), round(net, 2))
        ws.write('A43', 'Credit / debit notes are listed in the CDNR sheet. Check that they are accounted for in Books.', note)
        nxt = 45
    else:
        ws.write('A40', 'No CDNR (credit / debit note) data was provided.', note)
        nxt = 42

    # ---- E
    secrow(nxt, 'E. MATCHING RULES USED')
    rules = [
        '1. Key = GSTIN + invoice number (case, spaces and / - . _ ignored; leading zeros of every part ignored: 26-27/0110 = 26-27/110). Prefixes are not removed.',
        f"2. If still unmatched: same GSTIN + taxable and GST within Rs {res['tol']:g} + date gap up to {res['max_gap']} day(s) -> Inv No Mismatch (probable match). "
        'Books rows with blank GSTIN are matched on inv no + amounts.',
        '3. Same inv no + amounts but different GSTIN -> GSTIN Mismatch. Same GSTIN + inv no + amounts repeated in Books -> Duplicate Invoice.',
        '4. Priority per paired invoice: Previous Invoice (2B date before cutoff) > ITC Not Claimed (no GST in Books) > Inv No Mismatch > GSTIN Mismatch > Value Mismatch > Date Mismatch > Matched.',
        '5. Value Mismatch = taxable, IGST, CGST or SGST differs by more than the tolerance (head-wise; a missing head is also shown). Date Mismatch = everything matches but the date differs.',
        '6. Missing in 2B = Books entries with no 2B invoice. Missing in Tally = 2B invoices not found in Books (dated on/after the cutoff).',
    ]
    if settings.get('ignored_books') or settings.get('ignored_2b'):
        rules.append(f"7. Rows ignored (no GSTIN and no invoice no, e.g. blank / total rows): Books {settings.get('ignored_books', 0)}, 2B {settings.get('ignored_2b', 0)}.")
    for i, t in enumerate(rules):
        ws.write(nxt + i, 0, t, F())
    ws.freeze_panes(3, 0)
    ws.activate()
    ws.set_first_sheet()
    wb.close()
    out.seek(0)
    return out
