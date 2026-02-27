import pandas as pd
import pdfplumber
import io
import re
from datetime import datetime
from typing import List
from models import Transaction

# Exhaustive header aliases for both CSV and PDF
HEADER_MAPS = {
    'date': ['date', 'transaction date', 'time'],
    'description': ['description', 'transaction details', 'narration', 'details', 'name', 'particulars'],
    'credit': ['credit', 'money in', 'deposit', 'inward'],
    'debit': ['debit', 'money out', 'withdrawal', 'outward'],
    'amount': ['amount', 'value'],
    'category': ['category', 'type', 'class'],
    'ref': ['reference number', 'transaction reference', 'reference', 'ref', 'chq/ref', 'ref no', 'ref number', 'doc ref'],
    'currency': ['currency', 'ccy'],
    'balance': ['balance', 'running balance']
}

def clean_amount(val) -> float:
    if not val:
        return 0.0
    val_str = str(val).strip()
    if val_str.lower() == 'nan' or not val_str:
        return 0.0
    cleaned_str = re.sub(r'[^\d.-]', '', val_str)
    if not cleaned_str or cleaned_str == '-' or cleaned_str == '.':
        return 0.0
    try:
        return float(cleaned_str)
    except ValueError:
        return 0.0

def normalize_date(date_str: str) -> str:
    """Consistently format dates as YYYY-MM-DD for tool-based filtering."""
    if not date_str:
        return "Unknown Date"
    
    # Try common formats
    cleaned = re.sub(r'\s+', ' ', date_str.strip())
    
    # Handle formats like "16-Feb- 2026" or "16-feb-2026"
    # Improved regex to handle multiple separators and greedy year matching
    match = re.search(r'(\d{1,2})[-\s]([a-z]{3})[-\s]*(\d{2,4})?', cleaned, re.I)
    if match:
        day, month_str, year = match.groups()
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
            'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        mm = month_map.get(month_str.lower()[:3], '01')
        dd = day.zfill(2)
        
        # Default to current year if missing, or use provided year
        yy = year if year and len(year) >= 2 else str(datetime.now().year)
        if len(yy) == 2: yy = f"20{yy}"
        
        normalized = f"{yy}-{mm}-{dd}"
        print(f"DEBUG: Normalized '{cleaned}' to '{normalized}'")
        return normalized
        
    # Standard YYYY-MM-DD check
    match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', cleaned)
    if match:
        return f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
        
    return cleaned

def find_idx(headers, key):
    aliases = HEADER_MAPS.get(key, [])
    for i, h in enumerate(headers):
        h_clean = str(h).lower().replace('\n', ' ').strip()
        if h_clean in aliases:
            return i
        h_alpha = re.sub(r'[^a-z0-9 ]', ' ', h_clean).strip()
        h_alpha = re.sub(r' +', ' ', h_alpha)
        if h_alpha in aliases:
            return i
        for alias in aliases:
            if alias in h_alpha:
                return i
    return -1

def parse_csv(content: bytes) -> List[Transaction]:
    """Parses a CSV bank statement and returns a list of Transactions."""
    try:
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [str(c).lower().strip() for c in df.columns]
        mapping = {k: find_idx(df.columns, k) for k in HEADER_MAPS.keys()}
        
        if mapping['date'] == -1:
            print(f"DEBUG: No date column found in CSV. Columns: {df.columns.tolist()}")
            return []
        
        transactions = []
        for _, row in df.iterrows():
            try:
                date_val = normalize_date(str(row.iloc[mapping['date']]))
                
                desc_parts = []
                # description mapping combines several description-like fields
                if mapping['description'] != -1 and not pd.isna(row.iloc[mapping['description']]):
                    desc_parts.append(str(row.iloc[mapping['description']]).strip())
                    
                # Other individual fields
                for field in ['currency', 'balance']:
                    if mapping[field] != -1 and not pd.isna(row.iloc[mapping[field]]):
                        prefix = "Balance: " if field == 'balance' else ""
                        desc_parts.append(f"{prefix}{row.iloc[mapping[field]]}")
                
                desc_val = " - ".join(desc_parts) if desc_parts else "Unknown Description"
                
                amount_val = 0.0
                if mapping['credit'] != -1 and not pd.isna(row.iloc[mapping['credit']]):
                    amount_val += abs(clean_amount(row.iloc[mapping['credit']]))
                if mapping['debit'] != -1 and not pd.isna(row.iloc[mapping['debit']]):
                    amount_val -= abs(clean_amount(row.iloc[mapping['debit']]))
                
                if amount_val == 0.0 and mapping['amount'] != -1:
                    amount_val = clean_amount(row.iloc[mapping['amount']])
                    if mapping['debit'] != -1 and not pd.isna(row.iloc[mapping['debit']]) and amount_val > 0:
                        amount_val = -amount_val
                
                category_val = str(row.iloc[mapping['category']]) if mapping['category'] != -1 and not pd.isna(row.iloc[mapping['category']]) else "Uncategorized"
                
                transactions.append(Transaction(
                    date=date_val,
                    description=desc_val,
                    amount=amount_val,
                    category=category_val
                ))
            except Exception as e:
                continue
        return transactions
    except Exception as e:
        print(f"DEBUG: CSV Parse error: {e}")
        return []

def parse_pdf(content: bytes) -> List[Transaction]:
    """Robust PDF parser with multi-line row support."""
    transactions = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        print(f"DEBUG: PDF opened, pages: {len(pdf.pages)}")
        for page_num, page in enumerate(pdf.pages, 1):
            extracted_any_on_page = False
            for settings in [{}, {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_y_tolerance": 4}]:
                tables = page.extract_tables(table_settings=settings)
                if not tables: continue
                
                for table_num, table in enumerate(tables, 1):
                    if not table or len(table) < 2: continue
                    
                    header_mapping = None
                    header_row_idx = -1
                    for r_idx in range(min(15, len(table))):
                        potential_header = [str(h).lower().strip() if h else '' for h in table[r_idx]]
                        mapping = {k: find_idx(potential_header, k) for k in HEADER_MAPS.keys()}
                        if mapping['date'] != -1 and (mapping['amount'] != -1 or mapping['credit'] != -1 or mapping['debit'] != -1):
                            header_row_idx = r_idx
                            header_mapping = mapping
                            break
                    
                    if not header_mapping: continue

                    for row_idx, row in enumerate(table[header_row_idx + 1:], header_row_idx + 1):
                        if not row: continue
                        
                        all_text = " ".join([str(c) for c in row if c and str(c).strip()]).strip()
                        if not all_text: continue
                        
                        num_filled_cells = len([c for c in row if c and str(c).strip()])
                        if num_filled_cells <= 2 and "\n" in all_text:
                            # MERGED ROW LOGIC
                            lines = [line.strip() for line in all_text.split("\n") if line.strip()]
                            i = 0
                            while i < len(lines):
                                # Regex for date: e.g. 16-Feb or 16-02
                                date_match = re.search(r'(\d{1,2}-[a-z]{3}-?|\d{1,2}-\d{1,2}-?)', lines[i], re.I)
                                if date_match:
                                    date_part1 = date_match.group(1)
                                    # The rest of the line might be part of the description
                                    rem_of_line = lines[i].replace(date_part1, "").strip()
                                    
                                    meat_line = ""
                                    date_part2 = ""
                                    
                                    if i + 1 < len(lines):
                                        meat_line = lines[i+1]
                                        if rem_of_line:
                                            meat_line = f"{rem_of_line} {meat_line}"
                                            
                                        if i + 2 < len(lines) and re.match(r'^\d{4}$', lines[i+2]):
                                            date_part2 = lines[i+2]
                                            i += 3
                                        else:
                                            i += 2
                                        
                                        full_date = normalize_date(f"{date_part1} {date_part2}")
                                        amounts = re.findall(r'[\d,]+\.\d{2}', meat_line)
                                        if amounts:
                                            amt_val = clean_amount(amounts[0])
                                            # Improved sign detection: 'NIP FROM' is a credit, 'NIP TO' is a debit.
                                            is_debit = False
                                            meat_lower = meat_line.lower()
                                            
                                            # Specific debit keywords
                                            if any(k in meat_lower for k in ['transfer to', 'trf to', 'nip to', 'withdrawal', 'debit', 'fee', 'charge', 'comm', 'vat']):
                                                is_debit = True
                                            # Specific credit keywords (override debits if both present, e.g. "Transfer FROM")
                                            if any(k in meat_lower for k in ['receive', 'trf from', 'nip from', 'credit', 'deposit']):
                                                is_debit = False
                                            
                                            amt_val = -abs(amt_val) if is_debit else abs(amt_val)
                                            desc = meat_line
                                            
                                            # Be careful not to remove too much from description
                                            for a in amounts[:1]: 
                                                desc = desc.replace(a, "").strip()
                                            if len(amounts) >= 2: 
                                                # Use the last amount as balance if it matches common patterns
                                                desc = f"{desc} - Balance: {amounts[-1]}"
                                            
                                            transactions.append(Transaction(
                                                date=full_date,
                                                description=desc or "Unknown Description",
                                                amount=amt_val,
                                                category="Uncategorized"
                                            ))
                                    else: i += 1
                                else: i += 1
                            continue

                        # NORMAL ROW LOGIC
                        raw_date = str(row[header_mapping['date']]).replace('\n', ' ').strip()
                        date_val = normalize_date(raw_date)
                        if not date_val or date_val == "Unknown Date": continue
                        
                        desc_parts = []
                        m = header_mapping
                        for field in ['description', 'ref', 'currency', 'balance']:
                            if m[field] != -1 and row[m[field]]:
                                val = str(row[m[field]]).replace('\n', ' ').strip()
                                if val.lower() != 'nan':
                                    prefix = "Balance: " if field == 'balance' else ""
                                    desc_parts.append(f"{prefix}{val}")
                            
                        desc_val = " - ".join(desc_parts) if desc_parts else "Unknown Description"
                        amount_val = 0.0
                        if m['credit'] != -1 and row[m['credit']]: amount_val += abs(clean_amount(row[m['credit']]))
                        if m['debit'] != -1 and row[m['debit']]: amount_val -= abs(clean_amount(row[m['debit']]))
                        
                        if amount_val == 0.0 and m['amount'] != -1 and row[m['amount']]:
                            amount_val = clean_amount(row[m['amount']])
                            if m['debit'] != -1 and row[m['debit']] and amount_val > 0: amount_val = -amount_val
                        
                        category_val = "Uncategorized"
                        if m['category'] != -1 and row[m['category']]:
                            category_val = str(row[m['category']]).replace('\n', ' ').strip()
                        
                        transactions.append(Transaction(
                            date=date_val,
                            description=desc_val,
                            amount=amount_val,
                            category=category_val
                        ))
                        extracted_any_on_page = True
                if extracted_any_on_page: break
    print(f"DEBUG: PDF parsing complete. Found {len(transactions)} transaction(s)")
    return transactions
