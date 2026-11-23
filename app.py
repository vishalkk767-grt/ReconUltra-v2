"""
Recon Ultra (Powerful Reconciliation Platform)
-----------------------------------------------
Purchase Register (Books) vs GSTR-2B - 100% local.
Matching logic lives in recon_engine.py, this file is only the Streamlit UI.

Run:  streamlit run app.py
"""
import json
from datetime import date, datetime

import pandas as pd
import streamlit as st

from recon_engine import STATUS_ORDER, build_workbook, normalize, normalize_cdnr, reconcile

NONE = '-- None --'

# --------------------------------------------------------------------------
# File reading helpers
# --------------------------------------------------------------------------

def guess_column(cols, keywords):
    """First column whose name contains a keyword (keywords are tried in priority order)."""
    low = [str(c).lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(low):
            if kw in c:
                return cols[i]
    return None


def pick(label, cols, key, keywords, optional=False):
    options = ([NONE] if optional else []) + list(cols)
    guess = guess_column(cols, keywords)
    idx = options.index(guess) if guess in options else 0
    val = st.selectbox(label, options, index=idx, key=key)
    return None if val == NONE else val


BOOKS_SPEC = [
    ('gstin', 'Supplier GSTIN', ['gstin', 'ctin', 'gst no'], False),
    ('party', 'Party / Supplier Name', ['trade/legal', 'trade', 'party', 'particulars', 'supplier name', 'name'], True),
    ('invoice_no', 'Invoice / Voucher Number', ['invoice number', 'invoice no', 'inv no', 'voucher no', 'bill no', 'inum', 'invoice'], False),
    ('date', 'Invoice Date', ['invoice date', 'inv date', 'bill date', 'date', 'dt'], True),
    ('taxable', 'Taxable Value', ['taxable value', 'taxable', 'txval'], False),
    ('igst', 'IGST', ['igst', 'integrated', 'iamt'], True),
    ('cgst', 'CGST', ['cgst', 'central', 'camt'], True),
    ('sgst', 'SGST / UTGST', ['sgst', 'utgst', 'state/ut', 'state', 'samt'], True),
    ('gross', 'Gross Total / Invoice Value (optional - helps confirm GST not entered)', ['gross total', 'invoice value', 'bill amount', 'gross'], True),
]
G_SPEC = BOOKS_SPEC + [
    ('itc', 'ITC Availability (2B)', ['itc availability', 'itcavl', 'itc avail'], True),
]
CDNR_SPEC = [
    ('gstin', 'Supplier GSTIN', ['gstin', 'ctin'], False),
    ('party', 'Party Name', ['trade/legal', 'trade', 'party', 'name'], True),
    ('note_no', 'Note Number', ['note number', 'note no', 'ntnum', 'note'], True),
    ('note_type', 'Note Type (Credit / Debit)', ['note type', 'note typ', 'document type', 'typ'], True),
    ('date', 'Note Date', ['note date', 'date', 'dt'], True),
    ('taxable', 'Taxable Value', ['taxable value', 'taxable', 'txval'], True),
    ('igst', 'IGST', ['igst', 'integrated', 'iamt'], True),
    ('cgst', 'CGST', ['cgst', 'central', 'camt'], True),
    ('sgst', 'SGST / UTGST', ['sgst', 'utgst', 'state/ut', 'state', 'samt'], True),
]


def mapping_block(prefix, cols, spec, tag):
    return {k: pick(label, cols, f'{prefix}_{k}_{tag}', kws, optional=opt) for k, label, kws, opt in spec}


def read_table(file, sheets=None):
    """CSV -> DataFrame, Excel -> concat of chosen sheets."""
    if file.name.lower().endswith('.csv'):
        return pd.read_csv(file)
    xl = pd.ExcelFile(file)
    names = sheets or [xl.sheet_names[0]]
    frames = [xl.parse(s) for s in names]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _num(*vals):
    for v in vals:
        if v not in (None, ''):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _doc_amounts(doc):
    """GSTR-2B JSON: amounts sit on the document, or on its item list."""
    if any(k in doc for k in ('txval', 'igst', 'cgst', 'sgst')):
        return (_num(doc.get('txval')), _num(doc.get('igst')), _num(doc.get('cgst')), _num(doc.get('sgst')))
    items = doc.get('items') or doc.get('itms') or []
    tot = [0.0, 0.0, 0.0, 0.0]
    for it in items:
        it = it.get('itm_det', it)
        for i, k in enumerate(('txval', 'igst', 'cgst', 'sgst')):
            tot[i] += _num(it.get(k), it.get({'txval': 'txval', 'igst': 'iamt', 'cgst': 'camt', 'sgst': 'samt'}[k]))
    return tuple(tot)


def parse_gstr2b_json(file):
    """Returns (b2b DataFrame, cdnr DataFrame or None) from a GSTR-2B JSON download."""
    data = json.load(file)
    d = data.get('data', data)
    doc = d.get('docdata', d)
    b2b_rows, cdnr_rows = [], []
    for sup in doc.get('b2b', []) or []:
        for inv in sup.get('inv', []) or []:
            tx, ig, cg, sg = _doc_amounts(inv)
            b2b_rows.append({'GSTIN of Supplier': sup.get('ctin', ''), 'Party Name': sup.get('trdnm', '') or sup.get('cname', ''),
                             'Invoice Number': inv.get('inum', ''), 'Invoice Date': inv.get('dt', ''),
                             'Invoice Value': _num(inv.get('val')), 'Taxable Value': tx, 'IGST': ig, 'CGST': cg, 'SGST': sg,
                             'ITC Availability': {'Y': 'Yes', 'N': 'No'}.get(str(inv.get('itcavl', '')).upper(), '')})
    for sup in doc.get('cdnr', []) or []:
        for nt in sup.get('nt', []) or []:
            tx, ig, cg, sg = _doc_amounts(nt)
            cdnr_rows.append({'GSTIN of Supplier': sup.get('ctin', ''), 'Party Name': sup.get('trdnm', '') or sup.get('cname', ''),
                              'Note Number': nt.get('ntnum', ''), 'Note Type': nt.get('typ', '') or nt.get('ntty', ''),
                              'Note Date': nt.get('dt', ''), 'Taxable Value': tx, 'IGST': ig, 'CGST': cg, 'SGST': sg})
    if not b2b_rows:
        raise ValueError('No b2b invoices found in this JSON')
    return pd.DataFrame(b2b_rows), (pd.DataFrame(cdnr_rows) if cdnr_rows else None)


def itc_total(stats, keys):
    return sum(stats[k][f] for k in keys for f in ('ig', 'cg', 'sg'))


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title='Jarvis Ultra | Powerful Reconciliation Platform', page_icon='🤖',
                       layout='wide', initial_sidebar_state='expanded')
    st.markdown("""
    <style>
        .main-header {font-size: 2.4rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0.2rem;}
        .sub-header {font-size: 1.1rem; color: #5A6A7A; margin-bottom: 1.5rem;}
        .stDownloadButton button {background-color: #2563EB; color: white; border-radius: 8px;}
    </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('## 🤖 Jarvis Ultra')
        st.caption('(powerful reconciliation platform)')
        st.markdown('---')
        st.markdown('### Settings')
        amount_tol = st.number_input('Amount Tolerance (₹)', min_value=0.0, value=2.0, step=0.5,
                                     help='Taxable / GST difference up to this is treated as equal (rounding)')
        max_gap = st.number_input('Max Date Gap for fallback match (days)', min_value=0, max_value=30, value=2, step=1,
                                  help='When invoice numbers differ, match on amount only if the dates are within these days')
        cutoff = st.date_input('Previous Invoice cutoff', value=date(2026, 4, 1),
                               help='2B invoices dated BEFORE this date go to the Previous Invoice sheet')
        use_fuzzy = st.checkbox('Also try fuzzy invoice-no match (verify manually)', value=False,
                                help='Off by default: similar numbers like ...484 and ...494 can be different invoices')
        fuzzy_th = st.slider('Fuzzy threshold', 80, 100, 92, disabled=not use_fuzzy)
        st.markdown('---')
        st.info('🔒 100% Local • Your data never leaves your computer')

    st.markdown('<div class="main-header">🤖 Jarvis Ultra</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">(powerful reconciliation platform)</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(['🔄 Reconcile', '📖 Guide', 'ℹ️ About'])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader('1️⃣ Purchase Register (Books)')
            pr_file = st.file_uploader('Upload Excel / CSV', type=['xlsx', 'xls', 'csv'], key='pr')
        with c2:
            st.subheader('2️⃣ GSTR-2B')
            gstr_file = st.file_uploader('Upload Excel / CSV / JSON', type=['xlsx', 'xls', 'csv', 'json'], key='gstr')

        if pr_file and gstr_file:
            # ---- sheet selection --------------------------------------------------------
            pr_sheets, g_sheet, cd_sheet = None, None, None
            if not pr_file.name.lower().endswith('.csv'):
                names = pd.ExcelFile(pr_file).sheet_names
                pr_sheets = st.multiselect('Books sheet(s) to use (several sheets with same columns are combined)',
                                           names, default=[names[0]], key=f'prs_{pr_file.name}')
                if not pr_sheets:
                    st.warning('Select at least one Books sheet')
                    st.stop()
            cdnr_raw = None
            try:
                pr_raw = read_table(pr_file, pr_sheets)
            except Exception as e:
                st.error(f'Books file error: {e}')
                st.stop()
            gname = gstr_file.name.lower()
            try:
                if gname.endswith('.json'):
                    gstr_raw, cdnr_raw = parse_gstr2b_json(gstr_file)
                elif gname.endswith('.csv'):
                    gstr_raw = pd.read_csv(gstr_file)
                else:
                    gnames = pd.ExcelFile(gstr_file).sheet_names
                    d_main = next((i for i, n in enumerate(gnames) if ('b2b' in n.lower() or n.lower() in ('2b', 'gstr2b'))), 0)
                    cc1, cc2 = st.columns(2)
                    g_sheet = cc1.selectbox('GSTR-2B invoices sheet', gnames, index=d_main, key=f'gs_{gstr_file.name}')
                    cd_opts = [NONE] + gnames
                    d_cd = next((i + 1 for i, n in enumerate(gnames) if 'cdn' in n.lower()), 0)
                    cd_sheet = cc2.selectbox('Credit / Debit notes sheet (optional)', cd_opts, index=d_cd, key=f'cd_{gstr_file.name}')
                    gstr_raw = pd.read_excel(gstr_file, sheet_name=g_sheet)
                    if cd_sheet != NONE:
                        cdnr_raw = pd.read_excel(gstr_file, sheet_name=cd_sheet)
            except Exception as e:
                st.error(f'GSTR-2B file error: {e}')
                st.stop()
            gstr_raw = gstr_raw.dropna(how='all')
            pr_raw = pr_raw.dropna(how='all')
            st.success(f'Loaded → Books: {len(pr_raw)} rows | GSTR-2B: {len(gstr_raw)} rows'
                       + (f' | Credit/Debit notes: {len(cdnr_raw.dropna(how="all"))} rows' if cdnr_raw is not None else ''))

            # ---- column mapping -------------------------------------------------------
            st.markdown('### 3️⃣ Column Mapping')
            m1, m2 = st.columns(2)
            with m1:
                st.markdown('**Purchase Register columns**')
                pr_map = mapping_block('pr', list(pr_raw.columns), BOOKS_SPEC, pr_file.name)
            with m2:
                st.markdown('**GSTR-2B columns**')
                g_map = mapping_block('g', list(gstr_raw.columns), G_SPEC, gstr_file.name)
            cd_map = None
            if cdnr_raw is not None:
                with st.expander('Credit / Debit note columns'):
                    cd_map = mapping_block('cd', list(cdnr_raw.columns), CDNR_SPEC, gstr_file.name)

            if st.button('🚀 Run Advanced Reconciliation', type='primary', use_container_width=True):
                for nm, mp in (('Books', pr_map), ('GSTR-2B', g_map)):
                    used = [v for k, v in mp.items() if k in ('taxable', 'igst', 'cgst', 'sgst') and v]
                    if len(used) != len(set(used)):
                        st.warning(f'{nm}: same column selected for more than one amount field - please check mapping')
                with st.spinner('Jarvis Ultra is running reconciliation...'):
                    b_n, ib = normalize(pr_raw, pr_map)
                    g_n, ig = normalize(gstr_raw, g_map)
                    res = reconcile(b_n, g_n, tol=amount_tol, max_gap=int(max_gap), cutoff=pd.Timestamp(cutoff),
                                    use_fuzzy=use_fuzzy, fuzzy_th=fuzzy_th)
                    cdnr_n = normalize_cdnr(cdnr_raw, cd_map) if (cdnr_raw is not None and cd_map) else None
                    xl = build_workbook(res, cdnr_n, {'ignored_books': ib, 'ignored_2b': ig})
                    st.session_state.update({'res': res, 'xl': xl.getvalue(), 'ignored': (ib, ig), 'done': True})

            if st.session_state.get('done'):
                res, S = st.session_state['res'], st.session_state['res']['stats']
                ib, ig = st.session_state['ignored']
                st.markdown('---')
                st.subheader('📊 Reconciliation Summary Dashboard')
                if ib or ig:
                    st.info(f'Ignored rows (no GSTIN and no invoice no - blank / total rows): Books {ib}, 2B {ig}')

                k1, k2, k3, k4 = st.columns(4)
                tot = S['total_2b_itc']
                k1.metric('Total ITC as per 2B', f'₹ {tot:,.2f}')
                k2.metric('ITC matched', f"₹ {S['matched_itc']:,.2f}", f"{(S['matched_itc'] / tot * 100) if tot else 0:.1f}%")
                k3.metric('ITC remaining', f"₹ {S['remaining_itc']:,.2f}", f"{(S['remaining_itc'] / tot * 100) if tot else 0:.1f}%", delta_color='off')
                k4.metric('Books ITC missing in 2B', f"₹ {sum(S['Missing in 2B'][f] for f in ('bk_ig', 'bk_cg', 'bk_sg')):,.2f}")

                cnt = [('✅ Matched', S['Matched']['n']), ('📅 Date', S['Date Mismatch']['n']), ('⚠️ Value', S['Value Mismatch']['n']),
                       ('🟦 Inv No', S['Inv No Mismatch']['n']), ('🔄 GSTIN', S['GSTIN Mismatch']['n']),
                       ('♊ Duplicate', S['Duplicate Invoice']['n']), ('🚫 Not Claimed', S['ITC Not Claimed']['n']),
                       ('🕘 Previous', S['Previous Invoice']['n']), ('🔴 Missing 2B', S['Missing in 2B']['n']),
                       ('🟡 Missing Tally', S['Missing in Tally']['n'])]
                for col, (lbl, v) in zip(st.columns(len(cnt)), cnt):
                    col.metric(lbl, v)

                st.download_button('📥 Download Ultra Excel Report', data=st.session_state['xl'],
                                   file_name=f"Jarvis_Ultra_Reconciliation_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   use_container_width=True)

                r0, r1, r2, r3, r4, r5 = st.tabs(['📋 Result', '🔗 Matched Invoices', '🔴 Missing in 2B', '🟡 Missing in Tally',
                                                  '🕘 Previous Invoice', '🚫 ITC Not Claimed'])
                with r0:
                    rows = []
                    for lbl, k in (('Matched', 'Matched'), ('Date Mismatch', 'Date Mismatch'), ('Value Mismatch', 'Value Mismatch'),
                                   ('Inv No Mismatch', 'Inv No Mismatch'), ('GSTIN Mismatch', 'GSTIN Mismatch'),
                                   ('ITC Not Claimed', 'ITC Not Claimed'), ('Previous Invoice', 'Previous Invoice'),
                                   ('Missing in Tally', 'Missing in Tally')):
                        s = S[k]
                        rows.append({'Category': lbl, 'Invoices': s['n'], '2B IGST': s['ig'], '2B CGST': s['cg'], '2B SGST': s['sg'],
                                     '2B Total ITC': round(s['ig'] + s['cg'] + s['sg'], 2)})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption('Full Result sheet (formulas, Books side, credit notes, tie-out checks) is in the Excel download.')
                with r1:
                    m = res['matched']
                    if m.empty:
                        st.info('No matched invoices found.')
                    else:
                        sel = st.multiselect('Filter by status', STATUS_ORDER, default=[s for s in STATUS_ORDER if s != 'Matched'])
                        st.dataframe(m[m['Status'].isin(sel)] if sel else m, use_container_width=True, height=400)
                for tab, key, msg in ((r2, 'missing_2b', 'These invoices are in your books but missing in GSTR-2B.'),
                                      (r3, 'missing_tally', 'These appear in GSTR-2B but are not found in your books.'),
                                      (r4, 'previous', 'Invoices dated before the cutoff date.'),
                                      (r5, 'not_claimed', 'GST is in 2B but Books has no GST on the invoice.')):
                    with tab:
                        if res[key].empty:
                            st.success('Nothing here.')
                        else:
                            st.warning(msg)
                            st.dataframe(res[key], use_container_width=True, height=400)

    with tab2:
        st.markdown("""
        ### Jarvis Ultra Reconciliation Guide
        **Excel report mein sheets**
        1. **Result**: kitna ITC match hua / kitna baaki, category-wise (IGST/CGST/SGST), Books side, credit notes aur tie-out checks (formulas ke saath).
        2. **Matched Invoices**: Matched, Date Mismatch, Value Mismatch (head-wise Taxable/IGST/CGST/SGST difference), Inv No Mismatch, GSTIN Mismatch aur Duplicate Invoice - sab ek hi sheet par, `Status` column aur row colour ke saath.
        3. **Missing in 2B**: Books mein hai, 2B mein nahi.
        4. **Missing in Tally**: 2B mein hai, Books mein nahi.
        5. **Previous Invoice**: 2B invoice date cutoff (default 1 April 2026) se pehle ki.
        6. **ITC Not Claimed**: Books mein GST nahi, 2B mein GST hai.
        7. **CDNR**: credit / debit notes (agar upload kiye).

        **Matching order**: GSTIN + invoice no (leading zeros / `/ - .` ignore) → Books GSTIN blank ho to inv no + amount → same GSTIN + same amount + date gap ≤ Max Date Gap (invoice no alag) → same inv no + amount par GSTIN alag.
        Ek invoice sirf ek hi sheet mein aata hai. Priority: Previous Invoice > ITC Not Claimed > Inv No Mismatch > GSTIN Mismatch > Value Mismatch > Date Mismatch > Matched.
        """)
    with tab3:
        st.markdown('### About Jarvis Ultra')
        st.markdown('Matching engine: `recon_engine.py` • UI: `app.py`')

    st.markdown('---')
    st.caption('Jarvis Ultra • (powerful reconciliation platform)')


if __name__ == '__main__':
    main()
