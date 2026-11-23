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
  2b. 2B invoice dated BEFORE the cutoff, same GSTIN, amounts within tolerance, ANY date gap
                                               -> Late Booked (Books booked an old invoice late)
  3. Same invoice no, amounts within tolerance, GSTIN differs -> GSTIN Mismatch
  F. (optional) fuzzy invoice no, same GSTIN, date gap <= max_gap -> Fuzzy (verify)

Status given to every paired invoice (first rule that applies):
  Previous Invoice  -> 2B invoice date is before the cutoff date
  ITC Not Claimed   -> Books has no GST, 2B has GST (only if no_gst_as_value_mismatch=False; default: reported as Value Mismatch)
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


def _itc_flag(v):
    """2B 'ITC Availability' -> 'Yes' / 'No' / '' (blank)."""
    t = clean_text(v).upper()
    return 'No' if t.startswith('N') else 'Yes' if t.startswith('Y') else ''


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
    s = _col(df, m, 'gross')
    out['gross'] = s.map(clean_amount) if s is not None else np.nan
    s = _col(df, m, 'itc')
    out['itc'] = s.map(_itc_flag) if s is not None else ''
    out['tax'] = (out['igst'] + out['cgst'] + out['sgst']).round(2)
    out['gstin'] = out['gstin'].astype(str)      # keeps .str working on an empty sheet
    out['pan'] = out['gstin'].str[2:12]
    keep = ~((out['gstin'] == '') & (out['inv_raw'] == ''))
    ignored = int((~keep).sum())
    ign_tax = round(float(out.loc[~keep, 'tax'].sum()), 2)
    out = out[keep].reset_index(drop=True)
    out['rid'] = range(len(out))
    out.attrs['ignored_n'], out.attrs['ignored_tax'] = ignored, ign_tax
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


def reconcile(books, gst, tol=2.0, max_gap=2, cutoff='2026-04-01', use_fuzzy=False, fuzzy_th=92,
              no_gst_as_value_mismatch=True):
    """no_gst_as_value_mismatch=True: an invoice with GST in 2B but NO GST in Books is reported as
       Value Mismatch (with evidence in the remark) instead of 'ITC Not Claimed'."""
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

    # ---- Stage 2b: 2B invoice dated BEFORE cutoff, Books booked it late (any date gap) ----
    # Same GSTIN + taxable/GST within tolerance; invoice no and date may differ.
    ub_by, ug_by = {}, {}
    for r in free_books():
        if r['gstin'] != '' and r['tax'] > 0:
            ub_by.setdefault(r['gstin'], []).append(r)
    for r in free_gst():
        if not pd.isna(r['date']) and r['date'] < cutoff:
            ug_by.setdefault(r['gstin'], []).append(r)
    cands = []
    for gstin, brs in ub_by.items():
        for b in brs:
            for g in ug_by.get(gstin, []):
                if close(b, g):
                    cands.append((cost(b, g), b['rid'], g['rid']))
    for _, b, g in sorted(cands):
        if b in pairs or g in usedg:
            continue
        pair(b, g, 'Late Booked')

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
        if b['tax'] <= 0.005 and g['tax'] > 0.005 and not no_gst_as_value_mismatch:
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

    sup = {}        # supplier GSTIN -> [Books rows with GST, all Books rows]  (evidence for 'GST not entered')
    for r in B.values():
        if r['gstin']:
            x = sup.setdefault(r['gstin'], [0, 0])
            x[1] += 1
            x[0] += 1 if r['tax'] > 0.005 else 0

    def no_gst_evidence(b, g):
        """Evidence on whether GST was paid but not entered (amount booked = 2B invoice value) or never charged."""
        ev = []
        gb = b.get('gross', np.nan)
        booked = gb if (gb is not None and not pd.isna(gb) and gb > 0) else b['taxable']
        inv_val = g['taxable'] + g['tax']
        tl = max(tol, 1.0)
        if abs(booked - inv_val) <= tl:
            ev.append(f"Books amount {booked:,.2f} = 2B invoice value (taxable {g['taxable']:,.2f} + GST {g['tax']:,.2f}) -> "
                      'full bill incl. GST was booked without splitting GST: enter the GST ledger in Tally and claim ITC')
        elif abs(b['taxable'] - g['taxable']) <= tl:
            ev.append(f"Books taxable {b['taxable']:,.2f} = 2B taxable, no GST in Books -> GST may not have been paid; "
                      'get the supplier invoice and claim only if GST was really charged')
        else:
            ev.append(f"Books amount {booked:,.2f} differs from 2B invoice value {inv_val:,.2f} - check the bill")
        x = sup.get(b['gstin'])
        if x and x[1] > 1:
            ev.append(f"other Books entries of this supplier with GST: {x[0]} of {x[1] - 1}")
        return ev

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
            if b['tax'] <= 0.005 and g['tax'] > 0.005:
                ev = no_gst_evidence(b, g)
                return ('Value Mismatch: ' + '; '.join(parts) + ' | GST NOT ENTERED in Books? Confirm before claiming: '
                        + ('; '.join(ev) if ev else 'verify the supplier invoice; claim only if GST was charged and paid'))
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
            extra = ('; same GSTIN + amounts, inv no / date differ - probable late booking, verify'
                     if mt == 'Late Booked' else '')
            return (f"Invoice dated before {cutoff:%d-%m-%Y}; also in Books (voucher {b['inv_raw']}, "
                    f"Books date {_fmt(b['date'])}){extra}")
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
    bstat, gstat = {}, {}      # rid -> final status (written to the Books Data / 2B Data sheets)
    for b, (g, mt) in pairs.items():
        bb, gg = B[b], G[g]
        st = classify(bb, gg, mt)
        bstat[b], gstat[g] = st, st
        row = pair_row(bb, gg, st, mt, remark(st, bb, gg, mt))
        buckets['main' if st not in ('Previous Invoice', 'ITC Not Claimed') else st].append(row)
    for d, k in dupes.items():
        bb = B[d]
        bstat[d] = 'Duplicate Invoice'
        why = ('same GSTIN + inv no + amounts already matched with 2B' if k in pairs else
               'same GSTIN + inv no + amounts as another Books entry that is itself not in 2B')
        buckets['main'].append(pair_row(bb, None, 'Duplicate Invoice', f"Duplicate of {B[k]['inv_raw']}",
                                        f'Duplicate entry in Books ({why}) - ITC double-count risk'))
    # 2B-only rows
    missing_tally = []
    for g in free_gst():
        gstat[g['rid']] = 'Previous Invoice' if (not pd.isna(g['date']) and g['date'] < cutoff) else 'Missing in Tally'
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
        elif b['gstin'] == '':
            rem = 'GST booked but supplier GSTIN is blank in Books - add GSTIN in Tally, then check 2B'
        elif b['tax'] <= 0:
            rem = 'No GST in Books and invoice not in 2B'
        else:
            rem = 'Not in GSTR-2B - follow up with supplier (GSTR-1 not filed / late)'
        bstat[b['rid']] = 'Missing in 2B'
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
        'books_df': books, 'gst_df': gst, 'bstat': bstat, 'gstat': gstat,
        'ignored': {'books_n': books.attrs.get('ignored_n', 0), 'books_tax': books.attrs.get('ignored_tax', 0.0),
                    'gst_n': gst.attrs.get('ignored_n', 0), 'gst_tax': gst.attrs.get('ignored_tax', 0.0)},
        'control_2b': {'n': len(G), 'taxable': round(sum(r['taxable'] for r in G.values()), 2), 'igst': round(sum(r['igst'] for r in G.values()), 2),
                       'cgst': round(sum(r['cgst'] for r in G.values()), 2),
                       'sgst': round(sum(r['sgst'] for r in G.values()), 2)},
        'control_books': {'n': len(B), 'taxable': round(sum(r['taxable'] for r in B.values()), 2), 'igst': round(sum(r['igst'] for r in B.values()), 2),
                          'cgst': round(sum(r['cgst'] for r in B.values()), 2),
                          'sgst': round(sum(r['sgst'] for r in B.values()), 2)},
        'cutoff': cutoff, 'tol': tol, 'max_gap': max_gap,
        'no_gst_merged': no_gst_as_value_mismatch, 'paired_b': set(pairs), 'paired_g': {g for g, _ in pairs.values()}, 'dupes': set(dupes),
    }
    res['stats'] = compute_stats(res)
    res['audit_pre'] = pre_audit(books, gst, tol)
    res['audit_post'], res['pairs_verified'] = post_audit(res, tol)
    return res



# ======================================================================================
# AUDIT LAYER  (rule-based, deterministic - same input always gives the same findings)
#   pre_audit  : checks the raw Books / 2B rows BEFORE matching (data health)
#   post_audit : re-verifies the result AFTER matching with an independent recomputation
# ======================================================================================
GSTIN_RE = re.compile(r'^(0[1-9]|[12][0-9]|3[0-8]|97|99)[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$')
_RATES = [0.25, 1, 1.5, 3, 5, 6, 7.5, 12, 14, 18, 28, 40]

# id: (check name, severity, what it means / what to do)
CHECKS = {
    'P1': ('Invalid GSTIN format', 'High', 'GSTIN is not 15 characters / valid state code / PAN pattern. Correct it in the source file, else it cannot match.'),
    'P2': ('GST amount present but GSTIN blank', 'High', 'Add the supplier GSTIN in Tally; without it ITC cannot be matched reliably.'),
    'P3': ('Invoice number blank', 'High', 'Row cannot be matched on invoice no. Fill the voucher / invoice number.'),
    'P4': ('Invoice date missing or unreadable', 'High', 'Date could not be read; date checks and the cutoff rule are skipped for this row.'),
    'P5': ('Negative amount in a row', 'Medium', 'Credit-note style row inside the invoice list. Confirm it is intended.'),
    'P6': ('IGST together with CGST/SGST in one row', 'High', 'A supply is either inter-state (IGST) or intra-state (CGST+SGST). Check the tax heads.'),
    'P7': ('CGST and SGST are not equal', 'Medium', 'On an intra-state supply CGST = SGST. A difference usually means a keying error and shows up as Value Mismatch.'),
    'P8': ('Unusual GST rate (tax / taxable)', 'Low', 'Rate is not a standard slab. Normal for mixed-rate invoices; otherwise check taxable value / tax heads.'),
    'P9': ('Exact duplicate row inside the same file', 'High', 'Same GSTIN + invoice no + amounts appears more than once: risk of claiming ITC twice (Books) or portal duplicate (2B).'),
    'P10': ('Books supplier GSTIN not present anywhere in 2B', 'Medium', 'Supplier has not reported any invoice (not filed GSTR-1 / wrong GSTIN in Books). Follow up before claiming.'),
    'Q1': ('Pair re-verification failed', 'High', 'Independent recomputation disagrees with the status given by the matcher. Should always be zero.'),
    'Q2': ('Pair matched on weaker evidence (review)', 'Medium', 'Matched on GSTIN + amount (or blank GSTIN / late booking) while invoice no or date differs. Confirm once.'),
    'Q3': ('Possible missed match among unmatched rows', 'High', 'A Books row and a 2B row left unmatched look like the same invoice (same GSTIN / PAN and same amount). Check and link manually.'),
    'Q5': ('GST not entered in Books but 2B has GST (confirm before claiming)', 'High', 'Books shows no GST on an invoice where the supplier reported GST. Remark shows evidence (gross total, supplier pattern). Enter GST in Tally and claim, or confirm ITC is not due.'),
    'Q4': ('Books ITC higher than 2B on a paired invoice', 'High', 'Books claims more GST than the supplier reported. Excess ITC risk - claim only the 2B amount until resolved.'),
}
AUDIT_COLS = ['Phase', 'Check ID', 'Check', 'Source', 'Severity', 'GSTIN', 'Party', 'Invoice No (Books)',
              'Invoice No (2B)', 'Date', 'Taxable', 'Total GST', 'Note']


def pre_audit(books, gst, tol=2.0):
    """Data-health checks on the normalised Books / 2B frames, before matching."""
    out = []

    def add(cid, src, r, note=''):
        out.append(['Pre-match', cid, CHECKS[cid][0], src, CHECKS[cid][1], r['gstin'], r['party'],
                    r['inv_raw'] if src == 'Books' else '', r['inv_raw'] if src == '2B' else '',
                    r['date'], r['taxable'], r['tax'], note])

    g_set = set(gst['gstin'])
    for src, df in (('Books', books), ('2B', gst)):
        seen = set()
        for r in df.to_dict('records'):
            if r['gstin'] != '' and not GSTIN_RE.match(r['gstin']):
                add('P1', src, r, f"'{r['gstin']}' is not a valid GSTIN")
            if r['gstin'] == '' and r['tax'] > 0:
                add('P2', src, r, 'GST booked but GSTIN is blank')
            if r['inv_raw'] == '':
                add('P3', src, r)
            if pd.isna(r['date']):
                add('P4', src, r)
            if min(r['taxable'], r['igst'], r['cgst'], r['sgst']) < 0:
                add('P5', src, r)
            if r['igst'] > 0 and (r['cgst'] > 0 or r['sgst'] > 0):
                add('P6', src, r, f"IGST {r['igst']:,.2f} with CGST {r['cgst']:,.2f} / SGST {r['sgst']:,.2f}")
            if abs(r['cgst'] - r['sgst']) > tol:
                add('P7', src, r, f"CGST {r['cgst']:,.2f} vs SGST {r['sgst']:,.2f}")
            if r['taxable'] > 0 and r['tax'] > 0:
                rate = r['tax'] / r['taxable'] * 100
                if min(abs(rate - x) for x in _RATES) > 0.35:
                    add('P8', src, r, f'implied rate {rate:.2f}%')
            k = (r['gstin'], r['key'], round(r['taxable']), round(r['tax']))
            if k in seen and (r['gstin'] or r['inv_raw']):
                add('P9', src, r, 'same GSTIN + invoice no + amounts already present above')
            seen.add(k)
            if src == 'Books' and r['gstin'] != '' and r['tax'] > 0 and r['gstin'] not in g_set:
                add('P10', src, r, 'GSTIN has no invoice in 2B')
    return pd.DataFrame(out, columns=AUDIT_COLS)


def post_audit(res, tol=2.0):
    """Independent second opinion on the matching result (recomputed from the output rows)."""
    out, ver = [], {'n': 0}

    def add(cid, gstin, party, binv, ginv, date, taxable, tax, note, src='Books vs 2B'):
        out.append(['Post-match', cid, CHECKS[cid][0], src, CHECKS[cid][1], gstin, party, binv, ginv,
                    date, taxable, tax, note])

    T = tol + 0.005
    for df in (res['matched'], res['not_claimed'], res['previous']):
        for r in df.to_dict('records'):
            if str(r['Match Type']).startswith('Duplicate') or r['Invoice No (Books)'] == '' or r['Invoice No (2B)'] == '':
                continue
            ver['n'] += 1
            heads = [abs(r[f'{h} Diff (Books-2B)']) for h in ('Taxable', 'IGST', 'CGST', 'SGST')]
            maxd, st, mt = max(heads), r['Status'], r['Match Type']
            db, dg = r['Invoice Date (Books)'], r['Invoice Date (2B)']
            same_date = (not pd.isna(db)) and (not pd.isna(dg)) and db == dg
            bad = []
            if st == 'Matched' and not (maxd <= T and same_date):
                bad.append('Matched but amounts differ or date differs')
            if st == 'Date Mismatch' and not (maxd <= T and not same_date):
                bad.append('Date Mismatch but amounts differ or dates equal')
            if st == 'Value Mismatch' and not maxd > T:
                bad.append('Value Mismatch but all heads are within tolerance')
            if st in ('Inv No Mismatch', 'GSTIN Mismatch') and not maxd <= T:
                bad.append(f'{st} but amounts are not within tolerance')
            if mt in ('Exact', 'Normalized') and inv_key(r['Invoice No (Books)']) != inv_key(r['Invoice No (2B)']):
                bad.append('invoice no keys differ on an invoice-no match')
            if mt != 'Blank GSTIN' and st != 'GSTIN Mismatch' and r['GSTIN (Books)'] != r['GSTIN (2B)']:
                bad.append('GSTIN differs on a GSTIN match')
            for t in bad:
                add('Q1', r['GSTIN (2B)'], r['Party Name'], r['Invoice No (Books)'], r['Invoice No (2B)'],
                    r['Invoice Date (2B)'], r['Taxable (2B)'], r['Total GST (2B)'], t)
            if mt in ('Blank GSTIN', 'Near Match', 'Amount Fallback', 'Late Booked'):
                conf = 'Low' if mt == 'Late Booked' else 'Medium'
                add('Q2', r['GSTIN (2B)'], r['Party Name'], r['Invoice No (Books)'], r['Invoice No (2B)'],
                    r['Invoice Date (2B)'], r['Taxable (2B)'], r['Total GST (2B)'],
                    f"{mt} - confidence {conf}: Books date {_fmt(db)} vs 2B date {_fmt(dg)}")
            if st == 'Value Mismatch' and r['Total GST (Books)'] <= 0.005 and r['Total GST (2B)'] > 0.005:
                add('Q5', r['GSTIN (2B)'], r['Party Name'], r['Invoice No (Books)'], r['Invoice No (2B)'],
                    r['Invoice Date (2B)'], r['Taxable (2B)'], r['Total GST (2B)'], r['Remark'])
            if st in ('Value Mismatch', 'Inv No Mismatch', 'GSTIN Mismatch', 'Matched', 'Date Mismatch') \
                    and r['GST Diff (Books-2B)'] > T:
                add('Q4', r['GSTIN (2B)'], r['Party Name'], r['Invoice No (Books)'], r['Invoice No (2B)'],
                    r['Invoice Date (2B)'], r['Taxable (2B)'], r['Total GST (2B)'],
                    f"Books GST exceeds 2B by {r['GST Diff (Books-2B)']:,.2f}")

    # --- Q3: leftovers that still look like the same invoice -------------------------------
    pb, pg, dup = res['paired_b'], res['paired_g'], res['dupes']
    lb = [r for r in res['books_df'].to_dict('records') if r['rid'] not in pb and r['rid'] not in dup and r['tax'] > 0]
    lg = [r for r in res['gst_df'].to_dict('records') if r['rid'] not in pg and r['tax'] > 0]
    by_gstin, by_pan = {}, {}
    for r in lg:
        by_gstin.setdefault(r['gstin'], []).append(r)
        by_pan.setdefault(r['pan'], []).append(r)

    def near(b, g):
        return abs(b['tax'] - g['tax']) <= T and abs(b['taxable'] - g['taxable']) <= max(T, 0.01 * abs(g['taxable']))

    for b in lb:
        if b['gstin'] != '':
            cand = [(g, 'same GSTIN, same amount, invoice no / date differ') for g in by_gstin.get(b['gstin'], []) if near(b, g)]
            if not cand and b['pan']:
                cand = [(g, 'same PAN but different GSTIN, same amount') for g in by_pan.get(b['pan'], [])
                        if g['gstin'] != b['gstin'] and near(b, g)]
        else:
            cand = [(g, 'Books GSTIN blank, 2B row with same amount') for g in lg if near(b, g)]
        if cand:
            g, why = min(cand, key=lambda x: abs(b['tax'] - x[0]['tax']))
            add('Q3', g['gstin'], g['party'] or b['party'], b['inv_raw'], g['inv_raw'], b['date'], b['taxable'], b['tax'],
                f"{why}; 2B date {_fmt(g['date'])}, 2B GST {g['tax']:,.2f}" + (f" ({len(cand)} candidates)" if len(cand) > 1 else ''))
    return pd.DataFrame(out, columns=AUDIT_COLS), ver['n']


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

    # ---- Data sheets: the uploaded Books / 2B rows as read (Result control totals are formulas on these) ----
    DH = ['Row ID', 'Date', 'GSTIN', 'Party', 'Invoice No', 'Taxable', 'IGST', 'CGST', 'SGST', 'Total GST',
          'ITC Avl (2B)', 'Final Status']

    def data_sheet(name, df, stat):
        ws_ = wb.add_worksheet(name)
        ws_.set_row(0, 30)
        for j, (h, w) in enumerate(zip(DH, [8, 12, 18, 36, 22, 14, 13, 13, 13, 14, 11, 20])):
            ws_.write(0, j, h, HDR)
            ws_.set_column(j, j, w)
        for i, r in enumerate(df.to_dict('records'), start=1):
            ws_.write_number(i, 0, int(r['rid']) + 1, cell_fmt())
            put(ws_, i, 1, r['date'], cell_fmt(None, 'date'))
            put(ws_, i, 2, r['gstin'], cell_fmt())
            put(ws_, i, 3, r['party'], cell_fmt())
            put(ws_, i, 4, r['inv_raw'], cell_fmt())
            for j, k in ((5, 'taxable'), (6, 'igst'), (7, 'cgst'), (8, 'sgst')):
                ws_.write_number(i, j, float(r[k]), cell_fmt(None, 'num'))
            ws_.write_formula(i, 9, f'=G{i + 1}+H{i + 1}+I{i + 1}', cell_fmt(None, 'num'), float(r['tax']))
            put(ws_, i, 10, r.get('itc', ''), cell_fmt())
            put(ws_, i, 11, stat.get(r['rid'], ''), cell_fmt())
        ws_.autofilter(0, 0, max(len(df), 1), len(DH) - 1)
        ws_.freeze_panes(1, 0)

    data_sheet('Books Data', res['books_df'], res['bstat'])
    data_sheet('2B Data', res['gst_df'], res['gstat'])

    # ---- shared formats (used by the audit sheets and the Result sheet)
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

    # ---- Audit sheets: findings before matching (Pre) and the independent check after matching (Post) ----
    aud = pd.concat([res['audit_pre'], res['audit_post']], ignore_index=True)
    wd = wb.add_worksheet('Audit Details')
    wd.set_row(0, 30)
    for j, (h, w) in enumerate(zip(AUDIT_COLS, [11, 9, 38, 12, 9, 18, 34, 20, 20, 12, 14, 13, 70])):
        wd.write(0, j, h, HDR)
        wd.set_column(j, j, w)
    for i, r in enumerate(aud.to_dict('records'), start=1):
        for j, h in enumerate(AUDIT_COLS):
            v = r[h]
            if h == 'Date':
                put(wd, i, j, v, cell_fmt(None, 'date'))
            elif h in ('Taxable', 'Total GST'):
                put(wd, i, j, v, cell_fmt(None, 'num'))
            else:
                put(wd, i, j, v, cell_fmt())
    wd.autofilter(0, 0, max(len(aud), 1), len(AUDIT_COLS) - 1)
    wd.freeze_panes(1, 0)

    wa = wb.add_worksheet('Audit Summary')
    wa.set_column(0, 0, 11); wa.set_column(1, 1, 9); wa.set_column(2, 2, 46); wa.set_column(3, 3, 10)
    wa.set_column(4, 5, 13); wa.set_column(6, 6, 110)
    wa.write(0, 0, 'AUDIT - checks BEFORE matching (data health) and AFTER matching (independent verification)',
             F(bold=True, font_size=13))
    wa.write(1, 0, "Rule-based checks (same input = same findings). Counts are COUNTIFS formulas on the 'Audit Details' sheet; "
                   "filter that sheet by Check ID to see the rows.", note)
    for j, h in enumerate(['Phase', 'Check ID', 'Check', 'Severity', 'Rows flagged', 'Result', 'What it means / what to do']):
        wa.write(3, j, h, HDR)
    cnt = aud.groupby('Check ID').size().to_dict()
    for k, (cid, (nm, sev, why)) in enumerate(CHECKS.items()):
        rr = 4 + k
        wa.write(rr, 0, 'Pre-match' if cid[0] == 'P' else 'Post-match', cell_fmt())
        wa.write(rr, 1, cid, cell_fmt()); wa.write(rr, 2, nm, cell_fmt()); wa.write(rr, 3, sev, cell_fmt())
        wa.write_formula(rr, 4, f"=COUNTIFS('Audit Details'!$B:$B,B{rr + 1})", i_(), int(cnt.get(cid, 0)))
        wa.write_formula(rr, 5, f'=IF(E{rr + 1}>0,"REVIEW","OK")', cell_fmt(), 'REVIEW' if cnt.get(cid, 0) else 'OK')
        wa.write(rr, 6, why, F(font_size=9, font_color='#444444', text_wrap=True, valign='top'))
    rr = 4 + len(CHECKS) + 1
    wa.write(rr, 2, 'Pairs re-verified independently (Books row vs 2B row)', lab)
    wa.write_number(rr, 4, res['pairs_verified'], inpi)
    wa.write(rr + 1, 2, 'Failures in that re-verification (Q1, must be 0)', lab)
    wa.write_formula(rr + 1, 4, "=COUNTIFS('Audit Details'!$B:$B,\"Q1\")", i_(), int(cnt.get('Q1', 0)))

    # ---- Result sheet ----------------------------------------------------------------
    ws = ws_res
    S = res['stats']
    for col, w in zip('ABCDEFGHIJKL', [48, 20, 12, 17, 16, 15, 15, 17, 11, 18, 17, 95]):
        ws.set_column(f'{col}:{col}', w)
    ws.hide_gridlines(2)
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
    catrow(9, 'Value Mismatch (taxable / GST differs, incl. GST not entered in Books)', 'Matched Invoices', 'Value Mismatch', S['Value Mismatch'], 'status')
    catrow(10, 'Inv No Mismatch (probable match - verify)', 'Matched Invoices', 'Inv No Mismatch', S['Inv No Mismatch'], 'status')
    catrow(11, 'GSTIN Mismatch (inv no + amount same)', 'Matched Invoices', 'GSTIN Mismatch', S['GSTIN Mismatch'], 'status')
    catrow(12, ('ITC Not Claimed (now part of Value Mismatch - row 9)' if res.get('no_gst_merged') else 'ITC Not Claimed (no GST in Books, GST in 2B)'), 'ITC Not Claimed', None, S['ITC Not Claimed'], 'sheet')
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
    # control = totals of the '2B Data' sheet (the uploaded 2B file as read) -> formulas
    c2 = res['control_2b']
    G2 = "'2B Data'!"
    ws.write('A17', "Control: total of '2B Data' sheet (uploaded 2B file)", lab)
    ws.write_blank('B17', None, lab)
    ws.write_formula('C17', f'=COUNTA({G2}$A:$A)-1', i_(), c2['n'])
    ws.write_formula('D17', f'=SUM({G2}$F:$F)', n_(), c2['taxable'])
    ws.write_formula('E17', f'=SUM({G2}$G:$G)', n_(), c2['igst'])
    ws.write_formula('F17', f'=SUM({G2}$H:$H)', n_(), c2['cgst'])
    ws.write_formula('G17', f'=SUM({G2}$I:$I)', n_(), c2['sgst'])
    ws.write_formula('H17', '=E17+F17+G17', n_(), round(c2['igst'] + c2['cgst'] + c2['sgst'], 2))
    for col in 'IJK':
        ws.write_blank(f'{col}17', None, lab)
    ws.write('A18', 'Check: Total ITC as per 2B - Control (must be 0)', F(border=1, italic=True))
    ws.write_blank('B18', None, lab)
    for col, v in (('C', agg['n'] - c2['n']), ('D', agg['tx'] - c2['taxable']), ('E', agg['ig'] - c2['igst']),
                   ('F', agg['cg'] - c2['cgst']), ('G', agg['sg'] - c2['sgst']),
                   ('H', tot2b - (c2['igst'] + c2['cgst'] + c2['sgst']))):
        ws.write_formula(f'{col}18', f'={col}16-{col}17', n_() if col != 'C' else i_(), round(v, 2))
    for col in 'IJK':
        ws.write_blank(f'{col}18', None, lab)
    ws.write('A19', "Every figure on this sheet is a formula on the data sheets. Check rows must be 0: "
                    "no 2B / Books invoice is lost or counted twice.", note)

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
    B2 = "'Books Data'!"
    ws.write('A34', "Control: total of 'Books Data' sheet (uploaded Books file)", lab)
    for c in (1, 3, 8, 9, 10):
        ws.write_blank(33, c, None, lab)
    ws.write_formula('C34', f'=COUNTA({B2}$A:$A)-1', i_(), cb['n'])
    ws.write_formula('E34', f'=SUM({B2}$G:$G)', n_(), cb['igst'])
    ws.write_formula('F34', f'=SUM({B2}$H:$H)', n_(), cb['cgst'])
    ws.write_formula('G34', f'=SUM({B2}$I:$I)', n_(), cb['sgst'])
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
        '4. Priority per paired invoice: Previous Invoice (2B date before cutoff) > Inv No Mismatch > GSTIN Mismatch > Value Mismatch (incl. no GST in Books) > Date Mismatch > Matched.',
        '5. Value Mismatch = taxable, IGST, CGST or SGST differs by more than the tolerance (head-wise; a missing head is also shown). Date Mismatch = everything matches but the date differs.',
        '6. Missing in 2B = Books entries with no 2B invoice. Missing in Tally = 2B invoices not found in Books (dated on/after the cutoff).',
    ]
    rules += [
        f"7. Late Booked: a 2B invoice dated before {res['cutoff']:%d-%m-%Y} is paired with an unmatched Books entry of the same GSTIN "
        'when taxable and GST are within tolerance, whatever the invoice no / date. It stays in the Previous Invoice sheet (verify).',
        '8. Remarks: Duplicate Invoice says whether the original is matched with 2B or itself missing in 2B. '
        'Blank GSTIN with GST in Books -> "add GSTIN in Tally, then check 2B".',
    ]
    for i, t in enumerate(rules):
        ws.write(nxt + i, 0, t, F())

    # ---- F. data quality checks (formulas on the data sheets)
    r0 = nxt + len(rules) + 2
    secrow(r0, 'F. DATA QUALITY / EXTRA CHECKS')
    hdr(r0 + 1, {0: 'Particulars', 2: 'Count', 7: 'Total GST'})
    PVS = "'Previous Invoice'!"
    gdf = res['gst_df']
    itc_no = gdf[gdf['itc'] == 'No'] if 'itc' in gdf else gdf.iloc[0:0]
    late = res['previous'][res['previous']['Match Type'] == 'Late Booked']
    mm_ = res['matched']
    nogst = mm_[(mm_['Status'] == 'Value Mismatch') & (mm_['Total GST (Books)'] <= 0.005)]
    ig = res['ignored']
    frows = [
        ('2B invoices with ITC Availability = No (not eligible - reduce ITC)',
         f'=COUNTIF({G2}$K:$K,"No")', len(itc_no), f'=SUMIFS({G2}$J:$J,{G2}$K:$K,"No")', round(float(itc_no['tax'].sum()), 2)),
        ('Late Booked pairs (old 2B invoice, Books booked late - verify)',
         f'=COUNTIF({PVS}$D:$D,"Late Booked")', len(late),
         f'=SUMIFS({PVS}$W:$W,{PVS}$D:$D,"Late Booked")', round(float(late['Total GST (2B)'].sum()), 2)),
        ('Value Mismatch where Books has NO GST but 2B has GST - confirm before claiming (amount = 2B GST)',
         "=COUNTIFS('Matched Invoices'!$C:$C,\"Value Mismatch\",'Matched Invoices'!$V:$V,0)", len(nogst),
         "=SUMIFS('Matched Invoices'!$W:$W,'Matched Invoices'!$C:$C,\"Value Mismatch\",'Matched Invoices'!$V:$V,0)",
         round(float(nogst['Total GST (2B)'].sum()), 2)),
    ]
    for k, (lbl, fn, vn, fa, va) in enumerate(frows):
        r = r0 + 2 + k
        ws.write(r - 1, 0, lbl, lab)
        for c in (1, 3, 4, 5, 6, 8, 9, 10):
            ws.write_blank(r - 1, c, None, lab)
        ws.write_formula(r - 1, 2, fn, i_(), vn)
        ws.write_formula(r - 1, 7, fa, n_(), va)
    r = r0 + 2 + len(frows)
    for k, (lbl, n_ign, t_ign) in enumerate((('Rows ignored while reading Books (no GSTIN and no invoice no - blank / total rows)', ig['books_n'], ig['books_tax']),
                                              ('Rows ignored while reading 2B (no GSTIN and no invoice no - blank / total rows)', ig['gst_n'], ig['gst_tax']))):
        ws.write(r - 1 + k, 0, lbl, lab)
        for c in (1, 3, 4, 5, 6, 8, 9, 10):
            ws.write_blank(r - 1 + k, c, None, lab)
        ws.write_number(r - 1 + k, 2, n_ign, inpi)
        ws.write_number(r - 1 + k, 7, t_ign, inp)
    A_ = "'Audit Details'!"
    n_hi = int(((res['audit_pre']['Severity'] == 'High')).sum()); n_post = len(res['audit_post'])
    for k, (lbl, fn, vn) in enumerate((
            ("Pre-match audit: High-severity data issues (see 'Audit Summary')", f'=COUNTIFS({A_}$A:$A,"Pre-match",{A_}$E:$E,"High")', n_hi),
            ("Post-match audit: items to review (see 'Audit Summary')", f'=COUNTIFS({A_}$A:$A,"Post-match")', n_post))):
        ws.write(r + 1 + k, 0, lbl, lab)
        for c in (1, 3, 4, 5, 6, 7, 8, 9, 10):
            ws.write_blank(r + 1 + k, c, None, lab)
        ws.write_formula(r + 1 + k, 2, fn, i_(), vn)
    r += 2
    ws.write(r + 1, 0, 'Blue = counted while reading the file (not part of the data sheets). If the ignored GST equals a "Grand Total" line of your file, that is expected.', note)

    # ---- Logic column (what each Result row calculates)
    ws.write(4, 11, 'Logic / formula used', HDR)
    LOGIC = {
        6: "COUNTIF / SUMIFS on 'Matched Invoices' where Status = Matched. Amounts = 2B side (cols K, N, Q, T); Books ITC = col V.",
        7: "Same as row 6 for Status = Date Mismatch (everything equal, only invoice date differs).",
        8: "= Row 6 + Row 7  (invoice found in both Books and 2B, ITC can be taken).",
        9: "Status = Value Mismatch: taxable / IGST / CGST / SGST differs by more than the tolerance, head-wise. Includes invoices where Books has NO GST but 2B has GST (remark gives the evidence; see section F).",
        10: "Status = Inv No Mismatch: same GSTIN, amounts within tolerance and date gap within limit, but invoice no differs.",
        11: "Status = GSTIN Mismatch: same invoice no and amounts, supplier GSTIN differs.",
        12: ("Always 0 now: invoices with GST in 2B but none in Books are reported under Value Mismatch (row 9)." if res.get('no_gst_merged')
             else "SUM of the whole 'ITC Not Claimed' sheet: 2B has GST, Books has no GST on that invoice."),
        13: "SUM of the whole 'Previous Invoice' sheet: 2B invoice date is before the cutoff (includes Late Booked pairs).",
        14: "SUM of the whole 'Missing in Tally' sheet: 2B invoices not found in Books (dated on / after cutoff).",
        15: "= Rows 9 + 10 + 11 + 12 + 13 + 14  (ITC still to be resolved).",
        16: "= Row 8 + Row 15  (every 2B invoice sits in exactly one category).",
        17: "Totals of the '2B Data' sheet = the uploaded 2B file as read (COUNTA / SUM formulas).",
        18: "= Row 16 - Row 17. Must be 0 in every column; otherwise an invoice is lost or double counted.",
        23: "= Row 16 (total ITC as per 2B).",
        24: "= Row 8 (matched ITC).  % = row / Row 23.",
        25: "= Row 15 (ITC remaining).  % = row / Row 23.",
        30: "Books ITC of vouchers paired with a 2B invoice (Matched Invoices excl. duplicates + ITC Not Claimed + Previous Invoice).",
        31: "SUM of the 'Missing in 2B' sheet: Books vouchers with no 2B invoice (cannot be claimed until they appear in 2B).",
        32: "Duplicate Invoice rows in 'Matched Invoices' (same GSTIN + inv no + amounts booked more than once).",
        33: "= Row 30 + Row 31 + Row 32  (all Books vouchers).",
        34: "Totals of the 'Books Data' sheet = the uploaded Books file as read (COUNTA / SUM formulas).",
        35: "= Row 33 - Row 34. Must be 0 in every column.",
    }
    if cdnr is not None and len(cdnr):
        LOGIC.update({40: "COUNTIF / SUMIF on the CDNR sheet where Note Type = Credit Note (ITC to be reduced).",
                      41: "Same for Debit Note (ITC increase).", 42: "= Debit - Credit (net effect on ITC)."})
    lg = F(font_size=9, font_color='#444444', text_wrap=True, valign='top')
    for r, t in LOGIC.items():
        ws.write_string(r - 1, 11, t, lg)      # write_string: texts starting with '=' must not become formulas
    ws.freeze_panes(3, 0)
    ws.activate()
    ws.set_first_sheet()
    wb.close()
    out.seek(0)
    return out
