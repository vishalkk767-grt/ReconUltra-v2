# Jarvis Ultra - Purchase Register vs GSTR-2B

Files
- `app.py`           Streamlit UI (upload, sheet pick, column mapping, dashboard, download)
- `recon_engine.py`  Matching engine + Excel report (no Streamlit needed, easy to test)
- `requirements.txt`

Run
    pip install -r requirements.txt
    streamlit run app.py

Excel report sheets
- Result              : ITC matched / remaining (IGST/CGST/SGST), Books side, credit notes, tie-out checks (formulas)
- Matched Invoices    : Matched, Date Mismatch, Value Mismatch (head-wise), Inv No Mismatch, GSTIN Mismatch, Duplicate Invoice
- Missing in 2B       : in Books, not in 2B
- Missing in Tally    : in 2B, not in Books
- Previous Invoice    : 2B invoice date before cutoff (default 01-Apr-2026)
- ITC Not Claimed     : GST in 2B, no GST in Books
- CDNR                : credit / debit notes (if provided)

Use the engine without UI
    from recon_engine import normalize, reconcile, build_workbook
    b, _ = normalize(books_df, {'gstin':..,'party':..,'invoice_no':..,'date':..,'taxable':..,'igst':..,'cgst':..,'sgst':..})
    g, _ = normalize(gstr_df,  {...})
    res  = reconcile(b, g, tol=2.0, max_gap=2, cutoff='2026-04-01')
    open('out.xlsx','wb').write(build_workbook(res).getvalue())
