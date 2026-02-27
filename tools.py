from typing import List, Dict, Any
from collections import defaultdict, Counter
from statistics import mean, stdev
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from vdb import get_all_transactions, query_transactions

def get_spending_by_category(date_prefixes: list = None) -> str:
    """
    Returns a breakdown of spending (negative transactions) grouped by category.
    Use this to tell the user where their money is going overall.
    Optionally pass date_prefixes (list of YYYY or YYYY-MM) to filter by specific times.
    """
    transactions = get_all_transactions()
    category_totals = {}
    
    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
            
        if tx.amount < 0:
            cat = tx.category
            category_totals[cat] = category_totals.get(cat, 0.0) + tx.amount
            
    if not category_totals:
        return "No spending found."
        
    result = []
    for cat, total in category_totals.items():
        result.append(f"{cat}: ₦{abs(total):.2f}")
        
    return "\n".join(result)

def get_largest_expenses(limit: int = 5, date_prefixes: list = None) -> str:
    """
    Returns the largest negative transactions (expenses) in the provided limit.
    Use this to tell the user what their biggest individual purchases were.
    Optionally pass date_prefixes (list of YYYY or YYYY-MM) to filter by specific times.
    """
    try:
        if isinstance(limit, dict):
             # LLM hallucinated and gave us a dict
             limit = 5
        else:
             limit = int(limit)
    except (ValueError, TypeError):
        limit = 5
        
    transactions = get_all_transactions()
    
    # Filter for expenses and sort by amount ascending (largest negative first)
    expenses = []
    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        if tx.amount < 0:
            expenses.append(tx)
            
    expenses.sort(key=lambda tx: tx.amount)
    
    top_expenses = expenses[:min(limit, len(expenses))]
    
    if not top_expenses:
       return "No expenses found for the specified period."
       
    result = []
    for tx in top_expenses:
        result.append(f"{tx.date} - {tx.description}: ₦{abs(tx.amount):.2f} (Category: {tx.category})")
        
    return "\n".join(result)

def semantic_search_transactions(query: str, date_prefixes: list = None) -> str:
    """
    Looks up specific transactions matching a user's semantic query.
    For example: "Amazon purchases", "restaurants last month", "salary".
    Optionally filter by date_prefixes (list of YYYY or YYYY-MM) to filter by specific times.
    Returns results broken down by month with monthly totals.
    """
    results = query_transactions(query)

    # Apply date filtering if date_prefixes provided
    if date_prefixes and results:
        filtered = []
        for tx in results:
            tx_date = tx.date if tx.date else ""
            for prefix in date_prefixes:
                if tx_date.startswith(prefix):
                    filtered.append(tx)
                    break
        results = filtered

    if not results:
        date_info = f" for {date_prefixes}" if date_prefixes else ""
        return f"No transactions found matching '{query}'{date_info}"

    # Group transactions by month for breakdown
    monthly_groups = defaultdict(list)

    for tx in results:
        # Extract YYYY-MM from date
        if tx.date and len(tx.date) >= 7:
            month_key = tx.date[:7]  # YYYY-MM
        else:
            month_key = "Unknown"
        monthly_groups[month_key].append(tx)

    # Sort months chronologically
    sorted_months = sorted(monthly_groups.keys())

    formatted = []
    grand_total = 0.0

    for month in sorted_months:
        transactions = monthly_groups[month]
        month_total = sum(tx.amount for tx in transactions)
        grand_total += month_total

        # Format month header
        formatted.append(f"\n📅 {month}:")
        formatted.append(f"   Month Total: ₦{month_total:.2f}")
        formatted.append("   " + "-" * 50)

        # List transactions for this month
        for tx in transactions:
            formatted.append(f"   [{tx.date}] {tx.description} | {tx.category} | ₦{abs(tx.amount):.2f}")

    # Add summary
    formatted.insert(0, f"🔍 Found {len(results)} transaction(s) matching '{query}'")
    if date_prefixes:
        formatted.insert(1, f"📅 Filtered by: {date_prefixes}")
    formatted.insert(2, f"💰 Grand Total: ₦{grand_total:.2f}")
    formatted.insert(3, "")

    return "\n".join(formatted)

def get_total_credit_debit(date_prefixes: list = None) -> str:
    """
    Calculates the exact total amount of money in (credit) and money out (debit).
    Optionally pass date_prefixes (list of YYYY or YYYY-MM) to filter by specific times.
    """
    transactions = get_all_transactions()
    total_credit = 0.0
    total_debit = 0.0
    
    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
            
        if tx.amount > 0:
            total_credit += tx.amount
        elif tx.amount < 0:
            total_debit += tx.amount
            
    if total_credit == 0 and total_debit == 0:
        return "No transactions found for the specified period."
        
    return f"Total Credit (Money In): ₦{total_credit:.2f}\nTotal Debit (Money Out): ₦{abs(total_debit):.2f}"

def get_spending_by_description(query: str, date_prefixes: list = None) -> str:
    """
    Calculates the exact total amount of money sent to or received from a specific entity/person/description.
    query: A search string like the name of a person or business (e.g. "John", "Amazon")
    Optionally pass date_prefixes (list of YYYY or YYYY-MM) to filter by specific times.
    """
    transactions = get_all_transactions()
    total_credit = 0.0
    total_debit = 0.0
    found_count = 0
    query_lower = str(query).lower()
    
    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
            
        if query_lower in str(tx.description).lower() or query_lower in str(tx.category).lower():
            found_count += 1
            if tx.amount > 0:
                total_credit += tx.amount
            elif tx.amount < 0:
                total_debit += tx.amount
                
    if found_count == 0:
        return f"No transactions found matching '{query}' for the specified period."
        
    return f"Found {found_count} transactions matching '{query}'.\nTotal Received from '{query}': ₦{total_credit:.2f}\nTotal Sent to '{query}': ₦{abs(total_debit):.2f}"

def get_recipients(date_prefixes: list = None) -> str:
    """
    Returns a list of all names/entities the user has sent money to (debits/expenses).
    Optionally pass date_prefixes (list of YYYY or YYYY-MM) to filter by specific times.
    """
    transactions = get_all_transactions()
    recipients = {}
    
    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
            
        if tx.amount < 0:
            name = tx.description
            recipients[name] = recipients.get(name, 0.0) + tx.amount
            
    if not recipients:
        return "No outgoing transactions (recipients) found for the specified period."
        
    result = []
    # Sort by amount (most spent first)
    sorted_recipients = sorted(recipients.items(), key=lambda x: x[1])
    for name, total in sorted_recipients:
        result.append(f"{name}: ₦{abs(total):.2f}")
        
    return "List of recipients and total amounts sent:\n" + "\n".join(result)


def get_monthly_summary(date_prefixes: list = None) -> str:
    """
    Returns month-by-month breakdown of income, expenses, and net cash flow.
    Use this to show trends over multiple months or provide an overview of financial health by month.
    Optionally pass date_prefixes (list of YYYY or YYYY-MM) to filter by specific times.
    """
    transactions = get_all_transactions()
    monthly_data = defaultdict(lambda: {"income": 0.0, "expenses": 0.0})

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue

        month_key = tx.date[:7] if tx.date and len(tx.date) >= 7 else "Unknown"
        if tx.amount > 0:
            monthly_data[month_key]["income"] += tx.amount
        elif tx.amount < 0:
            monthly_data[month_key]["expenses"] += abs(tx.amount)

    if not monthly_data:
        return "No transactions found for the specified period."

    # Sort months chronologically
    sorted_months = sorted(monthly_data.keys())

    result = ["📊 Monthly Financial Summary", "=" * 60]
    result.append(f"{'Month':<12} {'Income':>15} {'Expenses':>15} {'Net Flow':>15}")
    result.append("-" * 60)

    total_income = 0.0
    total_expenses = 0.0

    for month in sorted_months:
        data = monthly_data[month]
        income = data["income"]
        expenses = data["expenses"]
        net_flow = income - expenses
        total_income += income
        total_expenses += expenses

        result.append(f"{month:<12} ₦{income:>13,.2f} ₦{expenses:>13,.2f} ₦{net_flow:>13,.2f}")

    result.append("-" * 60)
    total_net = total_income - total_expenses
    result.append(f"{'TOTAL':<12} ₦{total_income:>13,.2f} ₦{total_expenses:>13,.2f} ₦{total_net:>13,.2f}")

    return "\n".join(result)


def compare_periods(period1_prefixes: list, period2_prefixes: list) -> str:
    """
    Compare spending between two time periods (e.g., Jan vs Feb or Q1 vs Q2).
    Shows the difference in amount and percentage, with breakdown by category.
    period1_prefixes: List of date prefixes for first period (e.g., ['2025-01'])
    period2_prefixes: List of date prefixes for second period (e.g., ['2025-02'])
    """
    if not period1_prefixes or not period2_prefixes:
        return "Error: Both period prefixes are required."

    if not isinstance(period1_prefixes, list):
        period1_prefixes = [period1_prefixes]
    if not isinstance(period2_prefixes, list):
        period2_prefixes = [period2_prefixes]

    transactions = get_all_transactions()

    def calculate_period_data(prefixes):
        total_income = 0.0
        total_expenses = 0.0
        category_totals = defaultdict(float)

        for tx in transactions:
            if any(tx.date.startswith(str(p)) for p in prefixes):
                if tx.amount > 0:
                    total_income += tx.amount
                elif tx.amount < 0:
                    total_expenses += abs(tx.amount)
                    category_totals[tx.category] += abs(tx.amount)
        return total_income, total_expenses, category_totals

    income1, expenses1, cat1 = calculate_period_data(period1_prefixes)
    income2, expenses2, cat2 = calculate_period_data(period2_prefixes)

    if income1 == 0 and expenses1 == 0:
        return f"No transactions found for period 1: {period1_prefixes}"
    if income2 == 0 and expenses2 == 0:
        return f"No transactions found for period 2: {period2_prefixes}"

    # Calculate differences
    income_diff = income2 - income1
    expenses_diff = expenses2 - expenses1
    net1 = income1 - expenses1
    net2 = income2 - expenses2
    net_diff = net2 - net1

    def pct_change(old, new):
        if old == 0:
            return "N/A"
        return f"{((new - old) / old) * 100:+.1f}%"

    result = ["📊 Period Comparison", "=" * 60]
    result.append(f"Period 1: {period1_prefixes}")
    result.append(f"Period 2: {period2_prefixes}")
    result.append("")
    result.append(f"{'Metric':<20} {'Period 1':>15} {'Period 2':>15} {'Change':>15}")
    result.append("-" * 65)
    result.append(f"{'Income':<20} ₦{income1:>13,.2f} ₦{income2:>13,.2f} {pct_change(income1, income2):>15}")
    result.append(f"{'Expenses':<20} ₦{expenses1:>13,.2f} ₦{expenses2:>13,.2f} {pct_change(expenses1, expenses2):>15}")
    result.append(f"{'Net Cash Flow':<20} ₦{net1:>13,.2f} ₦{net2:>13,.2f} {pct_change(net1, net2):>15}")
    result.append("")
    result.append("📈 Category Breakdown (Period 2 vs Period 1)")
    result.append("-" * 65)

    all_categories = set(cat1.keys()) | set(cat2.keys())
    for cat in sorted(all_categories):
        v1 = cat1.get(cat, 0)
        v2 = cat2.get(cat, 0)
        diff = v2 - v1
        if v1 > 0 or v2 > 0:
            result.append(f"{cat:<20} ₦{v1:>13,.2f} → ₦{v2:>13,.2f} ({diff:+,.2f})")

    return "\n".join(result)


def get_income_by_source(date_prefixes: list = None) -> str:
    """
    Groups income transactions by source/description.
    Use this to show where money is coming from (salary, transfers, refunds, etc.).
    Optionally pass date_prefixes (list of YYYY or YYYY-MM) to filter by specific times.
    """
    transactions = get_all_transactions()
    source_totals = defaultdict(float)
    source_counts = defaultdict(int)

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue

        if tx.amount > 0:
            source_totals[tx.description] += tx.amount
            source_counts[tx.description] += 1

    if not source_totals:
        return "No income transactions found for the specified period."

    # Sort by amount (highest first)
    sorted_sources = sorted(source_totals.items(), key=lambda x: x[1], reverse=True)

    result = ["💰 Income by Source", "=" * 60]
    result.append(f"{'Source':<35} {'Count':>8} {'Total Amount':>15}")
    result.append("-" * 60)

    grand_total = 0.0
    for source, total in sorted_sources:
        count = source_counts[source]
        grand_total += total
        result.append(f"{source:<35} {count:>8} ₦{total:>13,.2f}")

    result.append("-" * 60)
    result.append(f"{'TOTAL':<35} {sum(source_counts.values()):>8} ₦{grand_total:>13,.2f}")

    return "\n".join(result)


def detect_anomalies(date_prefixes: list = None) -> str:
    """
    Finds unusual transactions based on statistical outliers (amount/frequency).
    Flags transactions that are more than 2 standard deviations from the mean.
    Also detects potential duplicate transactions (same amount/description on same day).
    """
    transactions = get_all_transactions()
    filtered_tx = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered_tx.append(tx)

    if not filtered_tx:
        return "No transactions found for the specified period."

    # Separate expenses and income
    expenses = [tx for tx in filtered_tx if tx.amount < 0]
    income = [tx for tx in filtered_tx if tx.amount > 0]

    result = ["🔍 Anomaly Detection Report", "=" * 60]
    anomalies_found = False

    # Statistical outliers for expenses
    if len(expenses) >= 2:
        expense_amounts = [abs(tx.amount) for tx in expenses]
        try:
            avg = mean(expense_amounts)
            std = stdev(expense_amounts) if len(expense_amounts) > 1 else 0
            threshold = avg + (2 * std)

            outliers = [tx for tx in expenses if abs(tx.amount) > threshold]
            if outliers:
                anomalies_found = True
                result.append(f"\n📊 Statistical Outliers (Expense > 2σ from mean)")
                result.append(f"   Average expense: ₦{avg:.2f}, Std Dev: ₦{std:.2f}")
                result.append(f"   Threshold (>₦{threshold:.2f}):")
                for tx in sorted(outliers, key=lambda x: x.amount):
                    result.append(f"   • {tx.date} | {tx.description} | ₦{abs(tx.amount):.2f}")
        except:
            pass

    # Statistical outliers for income
    if len(income) >= 2:
        income_amounts = [tx.amount for tx in income]
        try:
            avg = mean(income_amounts)
            std = stdev(income_amounts) if len(income_amounts) > 1 else 0
            threshold = avg + (2 * std)

            outliers = [tx for tx in income if tx.amount > threshold]
            if outliers:
                anomalies_found = True
                result.append(f"\n📊 Statistical Outliers (Income > 2σ from mean)")
                result.append(f"   Average income: ₦{avg:.2f}, Std Dev: ₦{std:.2f}")
                result.append(f"   Threshold (>₦{threshold:.2f}):")
                for tx in sorted(outliers, key=lambda x: x.amount, reverse=True):
                    result.append(f"   • {tx.date} | {tx.description} | ₦{tx.amount:.2f}")
        except:
            pass

    # Detect duplicates (same amount + description + date)
    seen = {}
    duplicates = []
    for tx in filtered_tx:
        key = (tx.date, tx.description, abs(tx.amount))
        if key in seen:
            duplicates.append((seen[key], tx))
        else:
            seen[key] = tx

    if duplicates:
        anomalies_found = True
        result.append(f"\n🔄 Potential Duplicate Transactions")
        for tx1, tx2 in duplicates:
            result.append(f"   • {tx1.date} | {tx1.description} | ₦{abs(tx1.amount):.2f}")
            result.append(f"     {tx2.date} | {tx2.description} | ₦{abs(tx2.amount):.2f}")

    if not anomalies_found:
        result.append("\n✅ No anomalies detected. All transactions appear normal.")

    return "\n".join(result)


def get_transaction_frequency(query: str, date_prefixes: list = None) -> str:
    """
    Shows how often money is spent at a specific merchant or for a query.
    Calculates the average days between transactions.
    """
    # Use semantic search to find matching transactions
    results = query_transactions(query)

    # Apply date filtering if provided
    if date_prefixes and results:
        filtered = []
        for tx in results:
            tx_date = tx.date if tx.date else ""
            for prefix in date_prefixes:
                if tx_date.startswith(prefix):
                    filtered.append(tx)
                    break
        results = filtered

    if not results:
        date_info = f" for {date_prefixes}" if date_prefixes else ""
        return f"No transactions found matching '{query}'{date_info}."

    # Filter to only expenses for frequency analysis
    expenses = [tx for tx in results if tx.amount < 0]

    if not expenses:
        return f"No expenses found matching '{query}'."

    # Count occurrences
    count = len(expenses)
    total_amount = sum(abs(tx.amount) for tx in expenses)
    avg_amount = total_amount / count if count > 0 else 0

    # Calculate frequency
    unique_dates = sorted(set(tx.date for tx in expenses if tx.date))

    result = [f"📊 Transaction Frequency Analysis: '{query}'", "=" * 60]
    result.append(f"Total Transactions: {count}")
    result.append(f"Total Amount Spent: ₦{total_amount:.2f}")
    result.append(f"Average Amount: ₦{avg_amount:.2f}")

    if len(unique_dates) > 1:
        # Calculate average days between transactions
        date_objects = []
        for date_str in unique_dates:
            try:
                date_objects.append(datetime.strptime(date_str, "%Y-%m-%d"))
            except:
                pass

        if len(date_objects) > 1:
            date_objects.sort()
            deltas = [(date_objects[i+1] - date_objects[i]).days for i in range(len(date_objects)-1)]
            avg_days = mean(deltas) if deltas else 0
            result.append(f"Average Days Between Transactions: {avg_days:.1f}")

            # First and last transaction
            result.append(f"\nFirst Transaction: {date_objects[0].strftime('%Y-%m-%d')}")
            result.append(f"Last Transaction: {date_objects[-1].strftime('%Y-%m-%d')}")
    elif unique_dates:
        result.append(f"\nAll transactions occurred on: {unique_dates[0]}")

    # Show breakdown by month
    monthly_counts = Counter(tx.date[:7] for tx in expenses if tx.date and len(tx.date) >= 7)
    if monthly_counts:
        result.append(f"\n📅 Monthly Breakdown:")
        for month in sorted(monthly_counts.keys()):
            result.append(f"   {month}: {monthly_counts[month]} transactions")

    return "\n".join(result)


def get_category_trend(category: str, months: int = 6) -> str:
    """
    Track spending in one specific category over time.
    Shows month-by-month spending with trend indicator (increasing/decreasing/stable).
    """
    if not category:
        return "Error: Category name is required."

    try:
        months = int(months) if isinstance(months, (int, float, str)) else 6
        if months < 1:
            months = 6
    except:
        months = 6

    transactions = get_all_transactions()
    category_lower = category.lower()

    # Group spending by month for the specified category
    monthly_spending = defaultdict(float)

    for tx in transactions:
        if tx.category.lower() == category_lower and tx.amount < 0:
            month_key = tx.date[:7] if tx.date and len(tx.date) >= 7 else None
            if month_key:
                monthly_spending[month_key] += abs(tx.amount)

    if not monthly_spending:
        return f"No spending found in category '{category}'."

    # Sort months and get the most recent 'months' count
    sorted_months = sorted(monthly_spending.keys())
    if len(sorted_months) > months:
        sorted_months = sorted_months[-months:]

    if len(sorted_months) < 2:
        return f"Category '{category}' - Only found data for {len(sorted_months)} month(s). Need at least 2 months for trend analysis."

    result = [f"📈 Category Trend: {category.title()}", "=" * 60]
    result.append(f"{'Month':<12} {'Amount':>15} {'Change':>15} {'Trend':<10}")
    result.append("-" * 55)

    total = 0.0
    prev_amount = None

    for month in sorted_months:
        amount = monthly_spending[month]
        total += amount

        if prev_amount is not None and prev_amount > 0:
            change = amount - prev_amount
            change_pct = (change / prev_amount) * 100
            change_str = f"{change:+.2f} ({change_pct:+.1f}%)"

            if change_pct > 10:
                trend = "📈 Rising"
            elif change_pct < -10:
                trend = "📉 Falling"
            else:
                trend = "➡️  Stable"
        else:
            change_str = "-"
            trend = "🆕 New"

        result.append(f"{month:<12} ₦{amount:>13,.2f} {change_str:>15} {trend:<10}")
        prev_amount = amount

    avg_monthly = total / len(sorted_months)
    result.append("-" * 55)
    result.append(f"{'Average':<12} ₦{avg_monthly:>13,.2f}")
    result.append(f"{'Total':<12} ₦{total:>13,.2f}")

    # Overall trend
    if len(sorted_months) >= 2:
        first = monthly_spending[sorted_months[0]]
        last = monthly_spending[sorted_months[-1]]
        if first > 0:
            overall_pct = ((last - first) / first) * 100
            result.append(f"\nOverall trend: {overall_pct:+.1f}% from {sorted_months[0]} to {sorted_months[-1]}")

    return "\n".join(result)


def get_transactions_by_date_range(start_date: str, end_date: str, query: str = "") -> str:
    """
    Filter transactions by exact date range with optional semantic query.
    start_date: Start date in YYYY-MM-DD format
    end_date: End date in YYYY-MM-DD format
    query: Optional search query to filter within results
    """
    if not start_date or not end_date:
        return "Error: Both start_date and end_date are required (YYYY-MM-DD format)."

    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Error: Dates must be in YYYY-MM-DD format (e.g., 2025-01-15)."

    if start_dt > end_dt:
        return "Error: Start date must be before or equal to end date."

    # Get transactions (use semantic search if query provided, else all)
    if query:
        transactions = query_transactions(query)
    else:
        transactions = get_all_transactions()

    # Filter by date range
    filtered = []
    for tx in transactions:
        if not tx.date:
            continue
        try:
            tx_dt = datetime.strptime(tx.date, "%Y-%m-%d")
            if start_dt <= tx_dt <= end_dt:
                filtered.append(tx)
        except:
            continue

    if not filtered:
        query_info = f" matching '{query}'" if query else ""
        return f"No transactions found between {start_date} and {end_date}{query_info}."

    # Sort by date
    filtered.sort(key=lambda x: x.date)

    result = [f"📅 Transactions from {start_date} to {end_date}", "=" * 60]
    if query:
        result.append(f"Query: '{query}'")
        result.append("")

    # Group by month for readability
    monthly_groups = defaultdict(list)
    for tx in filtered:
        month_key = tx.date[:7] if tx.date and len(tx.date) >= 7 else "Unknown"
        monthly_groups[month_key].append(tx)

    total_income = 0.0
    total_expenses = 0.0

    for month in sorted(monthly_groups.keys()):
        transactions = monthly_groups[month]
        result.append(f"\n📆 {month}:")

        for tx in transactions:
            if tx.amount > 0:
                total_income += tx.amount
                amount_str = f"+₦{tx.amount:.2f}"
            else:
                total_expenses += abs(tx.amount)
                amount_str = f"-₦{abs(tx.amount):.2f}"

            result.append(f"   [{tx.date}] {tx.description} | {tx.category} | {amount_str}")

    result.append("\n" + "=" * 60)
    result.append(f"Total Transactions: {len(filtered)}")
    result.append(f"Total Income: ₦{total_income:.2f}")
    result.append(f"Total Expenses: ₦{total_expenses:.2f}")
    result.append(f"Net: ₦{total_income - total_expenses:.2f}")

    return "\n".join(result)


def get_spending_velocity(date_prefixes: list = None, period: str = "daily") -> str:
    """
    Calculate daily or weekly average spending rate.
    Shows how quickly money is being spent.
    """
    if period not in ["daily", "weekly"]:
        period = "daily"

    transactions = get_all_transactions()
    filtered_expenses = []
    unique_dates = set()

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue

        if tx.amount < 0:
            filtered_expenses.append(tx)
            if tx.date:
                unique_dates.add(tx.date)

    if not filtered_expenses:
        return "No expenses found for the specified period."

    total_expenses = sum(abs(tx.amount) for tx in filtered_expenses)
    unique_days = len(unique_dates)

    # Parse dates to find actual date range
    date_objects = []
    for date_str in unique_dates:
        try:
            date_objects.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except:
            pass

    if len(date_objects) >= 2:
        date_range_days = (max(date_objects) - min(date_objects)).days + 1
        date_range_weeks = date_range_days / 7.0
    else:
        date_range_days = unique_days if unique_days > 0 else 1
        date_range_weeks = unique_days / 7.0 if unique_days > 0 else 1

    if date_range_days == 0:
        date_range_days = 1
    if date_range_weeks == 0:
        date_range_weeks = 1

    daily_avg = total_expenses / date_range_days
    weekly_avg = total_expenses / date_range_weeks

    # Transaction frequency
    tx_per_day = len(filtered_expenses) / date_range_days if date_range_days > 0 else 0

    result = [f"⚡ Spending Velocity Analysis", "=" * 60]
    result.append(f"Analysis Period: {period}")
    if date_prefixes:
        result.append(f"Date Filter: {date_prefixes}")
    result.append("")
    result.append(f"Total Expenses: ₦{total_expenses:.2f}")
    result.append(f"Transaction Count: {len(filtered_expenses)}")
    result.append(f"Unique Days with Spending: {unique_days}")
    result.append("")
    result.append(f"📊 Daily Rate:")
    result.append(f"   ₦{daily_avg:.2f} per day")
    result.append(f"   ~{tx_per_day:.1f} transactions per day")
    result.append("")
    result.append(f"📊 Weekly Rate:")
    result.append(f"   ₦{weekly_avg:.2f} per week")
    result.append(f"   ~{weekly_avg / 7:.2f} per day (if evenly distributed)")
    result.append("")

    # Projections
    result.append(f"📈 Projections:")
    result.append(f"   Monthly (30 days): ₦{daily_avg * 30:.2f}")
    result.append(f"   Yearly (365 days): ₦{daily_avg * 365:.2f}")

    return "\n".join(result)


def get_running_balance(date_prefixes: list = None) -> str:
    """
    Simulate account balance over time based on transaction amounts.
    Shows running total and identifies highest/lowest balance points.
    """
    transactions = get_all_transactions()
    filtered = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    if not filtered:
        return "No transactions found for the specified period."

    # Sort by date, then by amount (expenses before income on same day for realistic balance)
    filtered.sort(key=lambda x: (x.date, x.amount))

    running_total = 0.0
    min_balance = float('inf')
    max_balance = float('-inf')
    min_date = None
    max_date = None
    balance_points = []

    for tx in filtered:
        running_total += tx.amount
        balance_points.append((tx.date, tx.description, tx.amount, running_total))

        if running_total < min_balance:
            min_balance = running_total
            min_date = tx.date
        if running_total > max_balance:
            max_balance = running_total
            max_date = tx.date

    result = [f"💳 Running Balance Simulation", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    result.append(f"{'Date':<12} {'Description':<30} {'Amount':>12} {'Balance':>12}")
    result.append("-" * 68)

    # Show first 10 and last 10 if there are many transactions
    if len(balance_points) > 25:
        for date, desc, amount, balance in balance_points[:10]:
            amount_str = f"+₦{amount:.2f}" if amount > 0 else f"-₦{abs(amount):.2f}"
            desc_short = desc[:28] + ".." if len(desc) > 30 else desc
            result.append(f"{date:<12} {desc_short:<30} {amount_str:>12} ₦{balance:>10,.2f}")
        result.append(f"\n   ... {len(balance_points) - 20} transactions omitted ...\n")
        for date, desc, amount, balance in balance_points[-10:]:
            amount_str = f"+₦{amount:.2f}" if amount > 0 else f"-₦{abs(amount):.2f}"
            desc_short = desc[:28] + ".." if len(desc) > 30 else desc
            result.append(f"{date:<12} {desc_short:<30} {amount_str:>12} ₦{balance:>10,.2f}")
    else:
        for date, desc, amount, balance in balance_points:
            amount_str = f"+₦{amount:.2f}" if amount > 0 else f"-₦{abs(amount):.2f}"
            desc_short = desc[:28] + ".." if len(desc) > 30 else desc
            result.append(f"{date:<12} {desc_short:<30} {amount_str:>12} ₦{balance:>10,.2f}")

    result.append("-" * 68)
    result.append(f"\n📊 Balance Summary:")
    result.append(f"   Starting Balance: ₦0.00 (simulated)")
    result.append(f"   Ending Balance: ₦{running_total:,.2f}")
    result.append(f"   Highest Balance: ₦{max_balance:,.2f} on {max_date}")
    result.append(f"   Lowest Balance: ₦{min_balance:,.2f} on {min_date}")
    result.append(f"   Net Change: ₦{running_total:,.2f}")

    return "\n".join(result)


def get_day_of_week_analysis(date_prefixes: list = None) -> str:
    """
    Analyze spending patterns by day of week.
    Compares weekday (Mon-Fri) vs weekend (Sat-Sun) spending.
    """
    transactions = get_all_transactions()
    filtered = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    expenses = [tx for tx in filtered if tx.amount < 0]

    if not expenses:
        return "No expenses found for the specified period."

    weekday_totals = defaultdict(float)
    weekend_totals = defaultdict(float)
    weekday_count = 0
    weekend_count = 0
    weekday_amount = 0.0
    weekend_amount = 0.0

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_totals = defaultdict(float)
    daily_counts = defaultdict(int)

    for tx in expenses:
        try:
            dt = datetime.strptime(tx.date, "%Y-%m-%d")
            day_idx = dt.weekday()  # 0=Monday, 6=Sunday
            amount = abs(tx.amount)

            daily_totals[day_idx] += amount
            daily_counts[day_idx] += 1

            if day_idx < 5:  # Weekday
                weekday_amount += amount
                weekday_count += 1
            else:  # Weekend
                weekend_amount += amount
                weekend_count += 1
        except:
            continue

    result = ["📅 Day of Week Spending Analysis", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    # Daily breakdown
    result.append("Daily Breakdown:")
    result.append(f"{'Day':<15} {'Transactions':>12} {'Total Spent':>15} {'Average':>15}")
    result.append("-" * 60)

    for i, day_name in enumerate(day_names):
        total = daily_totals[i]
        count = daily_counts[i]
        avg = total / count if count > 0 else 0
        marker = "(Weekend)" if i >= 5 else ""
        result.append(f"{day_name:<15} {count:>12} ₦{total:>13,.2f} ₦{avg:>13,.2f} {marker}")

    result.append("-" * 60)

    # Weekday vs Weekend comparison
    result.append("\n📊 Weekday vs Weekend Comparison:")
    weekday_avg = weekday_amount / weekday_count if weekday_count > 0 else 0
    weekend_avg = weekend_amount / weekend_count if weekend_count > 0 else 0
    total = weekday_amount + weekend_amount

    result.append(f"{'':<15} {'Transactions':>12} {'Total Spent':>15} {'% of Total':>12}")
    result.append(f"{'Weekdays':<15} {weekday_count:>12} ₦{weekday_amount:>13,.2f} {(weekday_amount/total*100):>11.1f}%")
    result.append(f"{'Weekends':<15} {weekend_count:>12} ₦{weekend_amount:>13,.2f} {(weekend_amount/total*100):>11.1f}%")
    result.append("")
    result.append(f"Average per weekday transaction: ₦{weekday_avg:.2f}")
    result.append(f"Average per weekend transaction: ₦{weekend_avg:.2f}")

    if weekend_amount > 0:
        ratio = weekday_amount / weekend_amount
        result.append(f"\nWeekday/Weekend ratio: {ratio:.2f}x")
        if ratio > 2:
            result.append("💡 You spend significantly more on weekdays")
        elif ratio < 0.5:
            result.append("💡 You spend significantly more on weekends")
        else:
            result.append("💡 Your spending is relatively balanced")

    return "\n".join(result)


def get_time_of_month_analysis(date_prefixes: list = None) -> str:
    """
    Analyze spending patterns by time of month.
    Compares early month (1-10), mid-month (11-20), and end-of-month (21-31).
    """
    transactions = get_all_transactions()
    filtered = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    expenses = [tx for tx in filtered if tx.amount < 0]

    if not expenses:
        return "No expenses found for the specified period."

    # Define periods
    periods = {
        "Early Month (1-10)": (1, 10),
        "Mid Month (11-20)": (11, 20),
        "End of Month (21-31)": (21, 31)
    }

    period_data = {name: {"total": 0.0, "count": 0} for name in periods}

    for tx in expenses:
        try:
            day = int(tx.date.split("-")[2])
            amount = abs(tx.amount)

            for name, (start, end) in periods.items():
                if start <= day <= end:
                    period_data[name]["total"] += amount
                    period_data[name]["count"] += 1
                    break
        except:
            continue

    result = ["📅 Time of Month Spending Analysis", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    result.append(f"{'Period':<25} {'Transactions':>12} {'Total':>15} {'% of Total':>12}")
    result.append("-" * 65)

    grand_total = sum(d["total"] for d in period_data.values())

    for name, data in period_data.items():
        pct = (data["total"] / grand_total * 100) if grand_total > 0 else 0
        result.append(f"{name:<25} {data['count']:>12} ₦{data['total']:>13,.2f} {pct:>11.1f}%")

    result.append("-" * 65)
    total_count = sum(d["count"] for d in period_data.values())
    result.append(f"{'TOTAL':<25} {total_count:>12} ₦{grand_total:>13,.2f}")

    # Find highest spending period
    max_period = max(period_data.items(), key=lambda x: x[1]["total"])
    result.append(f"\n💡 Highest spending period: {max_period[0]} (₦{max_period[1]['total']:.2f})")

    return "\n".join(result)


def get_largest_expense_categories(limit: int = 5, date_prefixes: list = None) -> str:
    """
    Returns the largest expense categories with trend alerts.
    Shows top categories by total spending with month-over-month comparison.
    """
    try:
        limit = int(limit) if isinstance(limit, (int, float, str)) else 5
        if limit < 1:
            limit = 5
    except:
        limit = 5

    transactions = get_all_transactions()
    filtered = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    # Get current period data
    category_totals = defaultdict(float)
    category_monthly = defaultdict(lambda: defaultdict(float))

    for tx in filtered:
        if tx.amount < 0:
            amount = abs(tx.amount)
            category_totals[tx.category] += amount
            month = tx.date[:7] if tx.date and len(tx.date) >= 7 else "Unknown"
            category_monthly[tx.category][month] += amount

    if not category_totals:
        return "No expenses found for the specified period."

    # Sort by total spending
    sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    top_categories = sorted_categories[:limit]

    result = [f"📊 Top {limit} Expense Categories", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    result.append(f"{'Rank':<6} {'Category':<25} {'Total Spent':>15} {'% of Total':>12}")
    result.append("-" * 60)

    grand_total = sum(category_totals.values())

    for i, (category, total) in enumerate(top_categories, 1):
        pct = (total / grand_total * 100) if grand_total > 0 else 0
        result.append(f"{i:<6} {category:<25} ₦{total:>13,.2f} {pct:>11.1f}%")

    result.append("-" * 60)
    result.append(f"{'TOTAL':<32} ₦{grand_total:>13,.2f}")

    # Trend analysis for top categories
    result.append("\n📈 Trend Analysis (Last 3 Months):")

    for category, total in top_categories[:3]:  # Top 3 only
        monthly_data = category_monthly[category]
        if len(monthly_data) >= 2:
            sorted_months = sorted(monthly_data.keys())
            if len(sorted_months) >= 2:
                prev = monthly_data[sorted_months[-2]]
                curr = monthly_data[sorted_months[-1]]
                if prev > 0:
                    change = ((curr - prev) / prev) * 100
                    trend = "📈" if change > 5 else "📉" if change < -5 else "➡️"
                    result.append(f"   {category}: {change:+.1f}% {trend}")

    return "\n".join(result)


def find_similar_transactions(query: str, date_prefixes: list = None) -> str:
    """
    Find transactions similar to a query using semantic search.
    Groups related purchases and shows patterns.
    """
    results = query_transactions(query)

    # Apply date filtering
    if date_prefixes and results:
        filtered = []
        for tx in results:
            tx_date = tx.date if tx.date else ""
            for prefix in date_prefixes:
                if tx_date.startswith(prefix):
                    filtered.append(tx)
                    break
        results = filtered

    if not results:
        date_info = f" for {date_prefixes}" if date_prefixes else ""
        return f"No similar transactions found for '{query}'{date_info}."

    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # Cluster similar descriptions
    clusters = []
    used = set()

    for i, tx1 in enumerate(results):
        if i in used:
            continue
        cluster = [tx1]
        used.add(i)

        for j, tx2 in enumerate(results[i+1:], i+1):
            if j in used:
                continue
            if similarity(tx1.description, tx2.description) > 0.6:
                cluster.append(tx2)
                used.add(j)

        clusters.append(cluster)

    result = [f"🔍 Similar Transactions: '{query}'", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    total_all = sum(abs(tx.amount) for tx in results)
    result.append(f"Found {len(results)} related transaction(s), ₦{total_all:.2f} total")
    result.append("")

    # Show clusters
    for i, cluster in enumerate(clusters[:5], 1):  # Top 5 clusters
        cluster_total = sum(abs(tx.amount) for tx in cluster)
        descriptions = set(tx.description for tx in cluster)
        desc_str = ", ".join(descriptions)[:40]

        result.append(f"\nCluster {i}: {desc_str}{'...' if len(descriptions) > 1 else ''}")
        result.append(f"   {len(cluster)} transaction(s), ₦{cluster_total:.2f} total")

        # Show date range
        dates = [tx.date for tx in cluster if tx.date]
        if dates:
            result.append(f"   Dates: {min(dates)} to {max(dates)}")

    return "\n".join(result)


def get_merchant_spending(merchant: str, date_prefixes: list = None) -> str:
    """
    Get total spending at a specific merchant over time.
    Shows transaction history and monthly totals.
    """
    transactions = get_all_transactions()
    merchant_lower = merchant.lower()

    matching = []
    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue

        if merchant_lower in tx.description.lower():
            matching.append(tx)

    if not matching:
        return f"No transactions found for merchant '{merchant}'."

    # Separate income and expenses
    expenses = [tx for tx in matching if tx.amount < 0]
    income = [tx for tx in matching if tx.amount > 0]

    expense_total = sum(abs(tx.amount) for tx in expenses)
    income_total = sum(tx.amount for tx in income)
    net = income_total - expense_total

    # Group by month
    monthly_data = defaultdict(lambda: {"expenses": 0.0, "income": 0.0, "count": 0})
    for tx in matching:
        month = tx.date[:7] if tx.date and len(tx.date) >= 7 else "Unknown"
        monthly_data[month]["count"] += 1
        if tx.amount < 0:
            monthly_data[month]["expenses"] += abs(tx.amount)
        else:
            monthly_data[month]["income"] += tx.amount

    result = [f"🏪 Merchant Analysis: {merchant}", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    result.append("📊 Summary:")
    result.append(f"   Total Transactions: {len(matching)}")
    result.append(f"   Total Spent: ₦{expense_total:.2f}")
    if income_total > 0:
        result.append(f"   Total Received: ₦{income_total:.2f}")
        result.append(f"   Net: ₦{net:.2f}")

    result.append(f"\n📅 Monthly Breakdown:")
    result.append(f"{'Month':<12} {'Count':>8} {'Expenses':>15} {'Income':>15}")
    result.append("-" * 55)

    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        result.append(f"{month:<12} {data['count']:>8} ₦{data['expenses']:>13,.2f} ₦{data['income']:>13,.2f}")

    # Recent transactions
    result.append(f"\n📝 Recent Transactions:")
    recent = sorted(matching, key=lambda x: x.date, reverse=True)[:5]
    for tx in recent:
        amt_str = f"+₦{tx.amount:.2f}" if tx.amount > 0 else f"-₦{abs(tx.amount):.2f}"
        result.append(f"   {tx.date} | {tx.description} | {amt_str}")

    return "\n".join(result)


def get_top_merchants(limit: int = 10, date_prefixes: list = None) -> str:
    """
    Returns the most frequented or highest-spend merchants.
    Ranks by total spending and transaction count.
    """
    try:
        limit = int(limit) if isinstance(limit, (int, float, str)) else 10
        if limit < 1:
            limit = 10
    except:
        limit = 10

    transactions = get_all_transactions()
    filtered = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    expenses = [tx for tx in filtered if tx.amount < 0]

    if not expenses:
        return "No expenses found for the specified period."

    # Group by merchant (description)
    merchant_data = defaultdict(lambda: {"total": 0.0, "count": 0, "dates": set()})

    for tx in expenses:
        merchant_data[tx.description]["total"] += abs(tx.amount)
        merchant_data[tx.description]["count"] += 1
        merchant_data[tx.description]["dates"].add(tx.date)

    # Sort by total spent
    sorted_merchants = sorted(merchant_data.items(), key=lambda x: x[1]["total"], reverse=True)
    top_merchants = sorted_merchants[:limit]

    result = [f"🏪 Top {limit} Merchants by Spending", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    result.append(f"{'Rank':<6} {'Merchant':<30} {'Count':>8} {'Total':>15}")
    result.append("-" * 62)

    grand_total = sum(d["total"] for d in merchant_data.values())

    for i, (merchant, data) in enumerate(top_merchants, 1):
        pct = (data["total"] / grand_total * 100) if grand_total > 0 else 0
        merchant_short = merchant[:28] + ".." if len(merchant) > 30 else merchant
        result.append(f"{i:<6} {merchant_short:<30} {data['count']:>8} ₦{data['total']:>13,.2f}")

    result.append("-" * 62)
    result.append(f"{'TOTAL':<38} {sum(d['count'] for d in merchant_data.values()):>8} ₦{grand_total:>13,.2f}")

    # Most frequent
    most_frequent = max(merchant_data.items(), key=lambda x: x[1]["count"])
    result.append(f"\n💡 Most frequent merchant: {most_frequent[0]} ({most_frequent[1]['count']} visits)")

    return "\n".join(result)


def get_merchant_comparison(merchant1: str, merchant2: str, date_prefixes: list = None) -> str:
    """
    Compare spending between two merchants.
    Shows transaction counts, totals, and average transaction size.
    """
    if not merchant1 or not merchant2:
        return "Error: Both merchant names are required."

    transactions = get_all_transactions()
    filtered = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    m1_lower = merchant1.lower()
    m2_lower = merchant2.lower()

    m1_tx = [tx for tx in filtered if m1_lower in tx.description.lower() and tx.amount < 0]
    m2_tx = [tx for tx in filtered if m2_lower in tx.description.lower() and tx.amount < 0]

    if not m1_tx:
        return f"No transactions found for merchant '{merchant1}'."
    if not m2_tx:
        return f"No transactions found for merchant '{merchant2}'."

    m1_total = sum(abs(tx.amount) for tx in m1_tx)
    m2_total = sum(abs(tx.amount) for tx in m2_tx)
    m1_count = len(m1_tx)
    m2_count = len(m2_tx)
    m1_avg = m1_total / m1_count if m1_count > 0 else 0
    m2_avg = m2_total / m2_count if m2_count > 0 else 0

    result = [f"⚖️ Merchant Comparison", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    result.append(f"{'Metric':<25} {merchant1[:20]:<20} {merchant2[:20]:<20}")
    result.append("-" * 65)
    result.append(f"{'Transactions':<25} {m1_count:<20} {m2_count:<20}")
    result.append(f"{'Total Spent':<25} ₦{m1_total:<19,.2f} ₦{m2_total:<19,.2f}")
    result.append(f"{'Avg per Transaction':<25} ₦{m1_avg:<19,.2f} ₦{m2_avg:<19,.2f}")

    # Comparison
    result.append("")
    if m1_total > m2_total:
        diff = m1_total - m2_total
        pct = (diff / m2_total * 100) if m2_total > 0 else 0
        result.append(f"💡 You spent ₦{diff:.2f} more ({pct:.1f}%) at {merchant1}")
    elif m2_total > m1_total:
        diff = m2_total - m1_total
        pct = (diff / m1_total * 100) if m1_total > 0 else 0
        result.append(f"💡 You spent ₦{diff:.2f} more ({pct:.1f}%) at {merchant2}")
    else:
        result.append(f"💡 Spending is equal at both merchants")

    return "\n".join(result)


def detect_recurring_transactions(date_prefixes: list = None, min_occurrences: int = 3) -> str:
    """
    Identify subscriptions and recurring payments.
    Finds transactions with similar amounts on regular intervals.
    """
    try:
        min_occurrences = int(min_occurrences) if isinstance(min_occurrences, (int, float, str)) else 3
        if min_occurrences < 2:
            min_occurrences = 3
    except:
        min_occurrences = 3

    transactions = get_all_transactions()
    filtered = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    # Group by description and amount (within 5% tolerance)
    expense_groups = defaultdict(list)
    for tx in filtered:
        if tx.amount < 0:
            # Round amount to nearest 100 for grouping similar amounts
            key = (tx.description, round(abs(tx.amount), -2))
            expense_groups[key].append(tx)

    # Find recurring patterns
    recurring = []
    for (desc, amount), txs in expense_groups.items():
        if len(txs) >= min_occurrences:
            # Check if dates are roughly regular (monthly or weekly)
            dates = sorted([datetime.strptime(tx.date, "%Y-%m-%d") for tx in txs if tx.date])
            if len(dates) >= min_occurrences:
                # Calculate intervals
                intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                avg_interval = mean(intervals) if intervals else 0

                # If intervals are somewhat consistent (within 10 days of average)
                consistent = all(abs(i - avg_interval) <= 10 for i in intervals) if intervals else False

                if consistent or len(txs) >= 4:  # 4+ occurrences likely recurring
                    recurring.append({
                        "description": desc,
                        "amount": amount,
                        "count": len(txs),
                        "total": sum(abs(tx.amount) for tx in txs),
                        "avg_interval": avg_interval,
                        "first": dates[0].strftime("%Y-%m-%d"),
                        "last": dates[-1].strftime("%Y-%m-%d"),
                        "is_subscription": 25 <= avg_interval <= 35  # Monthly-ish
                    })

    if not recurring:
        return f"No recurring transactions detected (min {min_occurrences} occurrences)."

    # Sort by total spent
    recurring.sort(key=lambda x: x["total"], reverse=True)

    result = ["🔄 Recurring Transaction Detection", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    result.append(f"Found {len(recurring)} potential recurring payment(s)")
    result.append("")

    subscriptions = [r for r in recurring if r["is_subscription"]]
    others = [r for r in recurring if not r["is_subscription"]]

    if subscriptions:
        result.append("📅 Likely Subscriptions (Monthly):")
        result.append(f"{'Description':<30} {'Amount':>12} {'Count':>8} {'Total':>12}")
        result.append("-" * 65)
        sub_total = 0
        for r in subscriptions:
            result.append(f"{r['description'][:30]:<30} ₦{r['amount']:>10,.2f} {r['count']:>8} ₦{r['total']:>10,.2f}")
            sub_total += r['total']
        result.append("-" * 65)
        result.append(f"{'Monthly Subscription Total':<30} {'':>12} {'':>8} ₦{sub_total:>10,.2f}")
        result.append(f"{'Annual Projection':<30} {'':>12} {'':>8} ₦{sub_total*12:>10,.2f}")
        result.append("")

    if others:
        result.append("🔄 Other Recurring Patterns:")
        result.append(f"{'Description':<30} {'Amount':>12} {'Count':>8} {'Avg Days':>10}")
        result.append("-" * 65)
        for r in others:
            result.append(f"{r['description'][:30]:<30} ₦{r['amount']:>10,.2f} {r['count']:>8} {r['avg_interval']:>9.0f}")

    return "\n".join(result)


def get_subscription_summary(date_prefixes: list = None) -> str:
    """
    List all recurring charges with monthly total.
    Similar to detect_recurring but focused on subscription summary.
    """
    # Reuse detection logic but format as summary
    result_text = detect_recurring_transactions(date_prefixes, min_occurrences=2)

    if "No recurring" in result_text:
        return "No active subscriptions detected."

    # Parse and reformat as summary
    transactions = get_all_transactions()
    filtered = []

    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    # Find monthly subscriptions
    expense_groups = defaultdict(list)
    for tx in filtered:
        if tx.amount < 0:
            key = (tx.description, round(abs(tx.amount), -2))
            expense_groups[key].append(tx)

    monthly_subs = []
    for (desc, amount), txs in expense_groups.items():
        if len(txs) >= 2:
            dates = sorted([datetime.strptime(tx.date, "%Y-%m-%d") for tx in txs if tx.date])
            if len(dates) >= 2:
                intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                avg_interval = mean(intervals) if intervals else 0
                if 25 <= avg_interval <= 35:  # Monthly
                    monthly_subs.append({
                        "description": desc,
                        "amount": amount,
                        "next_expected": (dates[-1] + timedelta(days=30)).strftime("%Y-%m-%d")
                    })

    result = ["📋 Subscription Summary", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    if not monthly_subs:
        result.append("No monthly subscriptions detected.")
        return "\n".join(result)

    result.append(f"{'Service':<35} {'Monthly Cost':>15} {'Next Expected':>15}")
    result.append("-" * 68)

    total_monthly = 0
    for sub in sorted(monthly_subs, key=lambda x: x["amount"], reverse=True):
        desc_short = sub["description"][:33] + ".." if len(sub["description"]) > 35 else sub["description"]
        result.append(f"{desc_short:<35} ₦{sub['amount']:>13,.2f} {sub['next_expected']:>15}")
        total_monthly += sub["amount"]

    result.append("-" * 68)
    result.append(f"{'TOTAL MONTHLY':<35} ₦{total_monthly:>13,.2f}")
    result.append(f"{'ANNUAL COST':<35} ₦{total_monthly * 12:>13,.2f}")

    return "\n".join(result)


def get_upcoming_payments(days: int = 30, date_prefixes: list = None) -> str:
    """
    Predict upcoming recurring payments in the next N days.
    Based on detected recurring transaction patterns.
    """
    try:
        days = int(days) if isinstance(days, (int, float, str)) else 30
        if days < 1:
            days = 30
    except:
        days = 30

    transactions = get_all_transactions()
    today = datetime.now()
    cutoff = today + timedelta(days=days)

    filtered = []
    for tx in transactions:
        if date_prefixes:
            if not isinstance(date_prefixes, list):
                date_prefixes = [date_prefixes]
            if not any(tx.date.startswith(str(dp)) for dp in date_prefixes):
                continue
        filtered.append(tx)

    # Find recurring patterns
    expense_groups = defaultdict(list)
    for tx in filtered:
        if tx.amount < 0:
            key = (tx.description, round(abs(tx.amount), -2))
            expense_groups[key].append(tx)

    upcoming = []
    for (desc, amount), txs in expense_groups.items():
        if len(txs) >= 2:
            dates = sorted([datetime.strptime(tx.date, "%Y-%m-%d") for tx in txs if tx.date])
            if len(dates) >= 2:
                # Calculate next expected date
                intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                avg_interval = mean(intervals) if intervals else 30
                last_date = dates[-1]
                next_date = last_date + timedelta(days=int(avg_interval))

                if today <= next_date <= cutoff:
                    upcoming.append({
                        "description": desc,
                        "amount": amount,
                        "last_paid": last_date.strftime("%Y-%m-%d"),
                        "expected_date": next_date.strftime("%Y-%m-%d"),
                        "days_until": (next_date - today).days
                    })

    if not upcoming:
        return f"No upcoming recurring payments expected in the next {days} days."

    # Sort by expected date
    upcoming.sort(key=lambda x: x["expected_date"])

    result = [f"📅 Upcoming Payments (Next {days} Days)", "=" * 60]
    if date_prefixes:
        result.append(f"Period: {date_prefixes}")
        result.append("")

    result.append(f"{'Date':<12} {'Description':<30} {'Amount':>12} {'Status':<10}")
    result.append("-" * 68)

    total = 0
    for payment in upcoming:
        desc_short = payment["description"][:28] + ".." if len(payment["description"]) > 30 else payment["description"]
        status = "🔴 Due" if payment["days_until"] <= 3 else "🟡 Soon" if payment["days_until"] <= 7 else "🟢 Upcoming"
        result.append(f"{payment['expected_date']:<12} {desc_short:<30} ₦{payment['amount']:>10,.2f} {status:<10}")
        total += payment["amount"]

    result.append("-" * 68)
    result.append(f"{'TOTAL EXPECTED':<43} ₦{total:>10,.2f}")

    return "\n".join(result)


# A registry of tools that maps function names to actual callables
TOOL_REGISTRY = {
    "get_spending_by_category": get_spending_by_category,
    "get_largest_expenses": get_largest_expenses,
    "semantic_search_transactions": semantic_search_transactions,
    "get_total_credit_debit": get_total_credit_debit,
    "get_spending_by_description": get_spending_by_description,
    "get_recipients": get_recipients,
    # New tools - High Priority
    "get_monthly_summary": get_monthly_summary,
    "compare_periods": compare_periods,
    "get_income_by_source": get_income_by_source,
    # New tools - Medium Priority
    "detect_anomalies": detect_anomalies,
    "get_transaction_frequency": get_transaction_frequency,
    "get_category_trend": get_category_trend,
    # New tools - Advanced Filtering
    "get_transactions_by_date_range": get_transactions_by_date_range,
    "get_spending_velocity": get_spending_velocity,
    "get_running_balance": get_running_balance,
    # Pattern Analysis Tools
    "get_day_of_week_analysis": get_day_of_week_analysis,
    "get_time_of_month_analysis": get_time_of_month_analysis,
    "get_largest_expense_categories": get_largest_expense_categories,
    "find_similar_transactions": find_similar_transactions,
    # Merchant Analysis Tools
    "get_merchant_spending": get_merchant_spending,
    "get_top_merchants": get_top_merchants,
    "get_merchant_comparison": get_merchant_comparison,
    # Recurring/Subscription Tools
    "detect_recurring_transactions": detect_recurring_transactions,
    "get_subscription_summary": get_subscription_summary,
    "get_upcoming_payments": get_upcoming_payments
}
