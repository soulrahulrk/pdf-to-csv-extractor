"""
Built-in Document Types

Pre-configured document types for common document formats.
"""

from .document_type import (
    DocumentType,
    DocumentConfig,
    FieldDefinition,
    FieldType,
    ValidationRule,
    create_field,
)


# =============================================================================
# INVOICE TYPE
# =============================================================================

def _validate_invoice_totals(data: dict) -> list:
    """Cross-field validation for invoice totals."""
    errors = []
    
    subtotal = data.get('subtotal')
    tax = data.get('tax_amount')
    total = data.get('grand_total')
    
    if subtotal is not None and tax is not None and total is not None:
        try:
            subtotal_f = float(str(subtotal).replace(',', '').replace('$', ''))
            tax_f = float(str(tax).replace(',', '').replace('$', ''))
            total_f = float(str(total).replace(',', '').replace('$', ''))
            
            expected = subtotal_f + tax_f
            if abs(expected - total_f) > 0.02:  # Allow 2 cent tolerance
                errors.append(
                    f"Total mismatch: {subtotal_f} + {tax_f} = {expected}, "
                    f"but total is {total_f}"
                )
        except (ValueError, TypeError):
            pass
    
    return errors


INVOICE_TYPE = DocumentType(
    name='invoice',
    display_name='Invoice',
    description='Standard invoice with line items, totals, and payment details',
    version='1.0',
    
    fields=[
        # Header fields
        create_field(
            name='invoice_number',
            field_type=FieldType.IDENTIFIER,
            labels=['Invoice #', 'Invoice No', 'Invoice Number', 'Inv #', 'Invoice ID'],
            patterns=[
                r'INV[-/]?\d{4,}',
                r'#\s*\d{4,}',
                r'[A-Z]{2,3}[-/]?\d{4,}',
            ],
            required=True,
            group='header',
            description='Unique invoice identifier',
        ),
        create_field(
            name='invoice_date',
            field_type=FieldType.DATE,
            labels=['Invoice Date', 'Date', 'Issued', 'Issue Date'],
            patterns=[
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
                r'[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4}',
            ],
            required=True,
            group='header',
        ),
        create_field(
            name='due_date',
            field_type=FieldType.DATE,
            labels=['Due Date', 'Payment Due', 'Due By', 'Pay By'],
            patterns=[
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            ],
            group='header',
        ),
        create_field(
            name='po_number',
            field_type=FieldType.IDENTIFIER,
            labels=['PO #', 'PO Number', 'Purchase Order', 'P.O.'],
            patterns=[r'PO[-/]?\d{4,}'],
            group='header',
        ),
        
        # Vendor info
        create_field(
            name='vendor_name',
            field_type=FieldType.TEXT,
            labels=['From', 'Vendor', 'Seller', 'Bill From', 'Company'],
            position_hint='top-left',
            group='vendor',
        ),
        create_field(
            name='vendor_address',
            field_type=FieldType.ADDRESS,
            labels=['Address'],
            position_hint='top-left',
            group='vendor',
        ),
        
        # Customer info
        create_field(
            name='customer_name',
            field_type=FieldType.TEXT,
            labels=['Bill To', 'Customer', 'Client', 'Buyer', 'Ship To'],
            group='customer',
        ),
        create_field(
            name='customer_address',
            field_type=FieldType.ADDRESS,
            labels=['Address'],
            group='customer',
        ),
        
        # Totals
        create_field(
            name='subtotal',
            field_type=FieldType.CURRENCY,
            labels=['Subtotal', 'Sub-Total', 'Sub Total', 'Net Amount'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='totals',
        ),
        create_field(
            name='tax_rate',
            field_type=FieldType.PERCENTAGE,
            labels=['Tax Rate', 'Tax %', 'VAT Rate', 'GST Rate'],
            patterns=[r'\d{1,2}(?:\.\d{1,2})?\s*%'],
            group='totals',
        ),
        create_field(
            name='tax_amount',
            field_type=FieldType.CURRENCY,
            labels=['Tax', 'Tax Amount', 'VAT', 'GST', 'Sales Tax'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='totals',
        ),
        create_field(
            name='discount',
            field_type=FieldType.CURRENCY,
            labels=['Discount', 'Discount Amount'],
            patterns=[r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='totals',
        ),
        create_field(
            name='grand_total',
            field_type=FieldType.CURRENCY,
            labels=['Total', 'Grand Total', 'Amount Due', 'Total Due', 'Balance Due'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            required=True,
            group='totals',
        ),
        
        # Payment
        create_field(
            name='payment_terms',
            field_type=FieldType.TEXT,
            labels=['Payment Terms', 'Terms', 'Net 30', 'Due Upon Receipt'],
            patterns=[r'Net\s*\d{1,3}', r'Due\s*(?:Upon|On)\s*Receipt'],
            group='payment',
        ),
        
        # Line items (table field)
        create_field(
            name='line_items',
            field_type=FieldType.TABLE,
            labels=['Description', 'Item', 'Product', 'Service'],
            group='items',
        ),
    ],
    
    identification_patterns=[
        r'invoice',
        r'inv\s*#',
        r'bill\s*to',
        r'amount\s*due',
    ],
    identification_keywords=[
        'invoice', 'bill to', 'ship to', 'amount due',
        'subtotal', 'tax', 'total', 'payment terms',
    ],
    
    table_fields=['line_items'],
    table_column_mappings={
        'line_items': [
            'description', 'item', 'product',
            'quantity', 'qty',
            'unit price', 'rate', 'price',
            'amount', 'total', 'line total',
        ],
    },
    
    cross_field_rules=[_validate_invoice_totals],
    
    category='financial',
    tags=['invoice', 'billing', 'payment'],
)


# =============================================================================
# BANK STATEMENT TYPE
# =============================================================================

def _validate_statement_balance(data: dict) -> list:
    """Validate statement balance calculations."""
    errors = []
    
    opening = data.get('opening_balance')
    closing = data.get('closing_balance')
    total_credits = data.get('total_credits')
    total_debits = data.get('total_debits')
    
    if all(v is not None for v in [opening, closing, total_credits, total_debits]):
        try:
            opening_f = float(str(opening).replace(',', '').replace('$', ''))
            closing_f = float(str(closing).replace(',', '').replace('$', ''))
            credits_f = float(str(total_credits).replace(',', '').replace('$', ''))
            debits_f = float(str(total_debits).replace(',', '').replace('$', ''))
            
            expected = opening_f + credits_f - debits_f
            if abs(expected - closing_f) > 0.02:
                errors.append(
                    f"Balance mismatch: {opening_f} + {credits_f} - {debits_f} = {expected}, "
                    f"but closing balance is {closing_f}"
                )
        except (ValueError, TypeError):
            pass
    
    return errors


BANK_STATEMENT_TYPE = DocumentType(
    name='bank_statement',
    display_name='Bank Statement',
    description='Monthly bank account statement with transactions and balances',
    version='1.0',
    
    fields=[
        # Account info
        create_field(
            name='account_number',
            field_type=FieldType.IDENTIFIER,
            labels=['Account Number', 'Account #', 'Account No', 'Acct #'],
            patterns=[
                r'\d{8,16}',
                r'[X*]+\d{4}',  # Masked account
            ],
            required=True,
            group='account',
        ),
        create_field(
            name='account_holder',
            field_type=FieldType.TEXT,
            labels=['Account Holder', 'Account Name', 'Name'],
            group='account',
        ),
        create_field(
            name='account_type',
            field_type=FieldType.TEXT,
            labels=['Account Type', 'Type'],
            patterns=[r'Checking|Savings|Money Market|CD'],
            group='account',
        ),
        
        # Statement period
        create_field(
            name='statement_date',
            field_type=FieldType.DATE,
            labels=['Statement Date', 'As Of', 'Date'],
            patterns=[r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'],
            required=True,
            group='period',
        ),
        create_field(
            name='period_start',
            field_type=FieldType.DATE,
            labels=['Period Start', 'From', 'Beginning'],
            patterns=[r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'],
            group='period',
        ),
        create_field(
            name='period_end',
            field_type=FieldType.DATE,
            labels=['Period End', 'To', 'Through', 'Ending'],
            patterns=[r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'],
            group='period',
        ),
        
        # Balances
        create_field(
            name='opening_balance',
            field_type=FieldType.CURRENCY,
            labels=['Opening Balance', 'Beginning Balance', 'Previous Balance'],
            patterns=[r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='balances',
        ),
        create_field(
            name='closing_balance',
            field_type=FieldType.CURRENCY,
            labels=['Closing Balance', 'Ending Balance', 'Current Balance', 'New Balance'],
            patterns=[r'-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            required=True,
            group='balances',
        ),
        create_field(
            name='available_balance',
            field_type=FieldType.CURRENCY,
            labels=['Available Balance', 'Available'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='balances',
        ),
        
        # Summary
        create_field(
            name='total_credits',
            field_type=FieldType.CURRENCY,
            labels=['Total Credits', 'Total Deposits', 'Credits'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='summary',
        ),
        create_field(
            name='total_debits',
            field_type=FieldType.CURRENCY,
            labels=['Total Debits', 'Total Withdrawals', 'Debits'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='summary',
        ),
        create_field(
            name='total_fees',
            field_type=FieldType.CURRENCY,
            labels=['Total Fees', 'Fees', 'Service Charges'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='summary',
        ),
        
        # Transactions table
        create_field(
            name='transactions',
            field_type=FieldType.TABLE,
            labels=['Transaction', 'Description', 'Activity'],
            group='transactions',
        ),
    ],
    
    identification_patterns=[
        r'bank\s*statement',
        r'account\s*statement',
        r'statement\s*of\s*account',
        r'checking|savings',
    ],
    identification_keywords=[
        'bank statement', 'account statement', 'checking', 'savings',
        'opening balance', 'closing balance', 'deposits', 'withdrawals',
    ],
    
    table_fields=['transactions'],
    table_column_mappings={
        'transactions': [
            'date', 'transaction date',
            'description', 'details', 'memo',
            'debit', 'withdrawal', 'credit', 'deposit',
            'amount', 'balance', 'running balance',
        ],
    },
    
    cross_field_rules=[_validate_statement_balance],
    
    category='financial',
    tags=['bank', 'statement', 'account', 'transactions'],
)


# =============================================================================
# RECEIPT TYPE
# =============================================================================

RECEIPT_TYPE = DocumentType(
    name='receipt',
    display_name='Receipt',
    description='Purchase receipt from retail or service transaction',
    version='1.0',
    
    fields=[
        # Store info
        create_field(
            name='store_name',
            field_type=FieldType.TEXT,
            labels=['Store', 'Merchant', 'Vendor'],
            position_hint='top',
            group='store',
        ),
        create_field(
            name='store_address',
            field_type=FieldType.ADDRESS,
            labels=['Address', 'Location'],
            position_hint='top',
            group='store',
        ),
        create_field(
            name='store_phone',
            field_type=FieldType.PHONE,
            labels=['Phone', 'Tel'],
            patterns=[r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'],
            group='store',
        ),
        
        # Transaction info
        create_field(
            name='receipt_number',
            field_type=FieldType.IDENTIFIER,
            labels=['Receipt #', 'Trans #', 'Transaction', 'Ref #'],
            patterns=[r'#?\d{6,}'],
            group='transaction',
        ),
        create_field(
            name='transaction_date',
            field_type=FieldType.DATE,
            labels=['Date', 'Transaction Date'],
            patterns=[r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'],
            required=True,
            group='transaction',
        ),
        create_field(
            name='transaction_time',
            field_type=FieldType.TEXT,
            labels=['Time'],
            patterns=[r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?'],
            group='transaction',
        ),
        create_field(
            name='cashier',
            field_type=FieldType.TEXT,
            labels=['Cashier', 'Clerk', 'Server', 'Associate'],
            group='transaction',
        ),
        
        # Payment
        create_field(
            name='payment_method',
            field_type=FieldType.TEXT,
            labels=['Payment', 'Paid By', 'Card Type', 'Tender'],
            patterns=[r'Cash|Credit|Debit|Visa|MC|Mastercard|Amex|Discover'],
            group='payment',
        ),
        create_field(
            name='card_last_four',
            field_type=FieldType.TEXT,
            labels=['Card #', 'Account'],
            patterns=[r'[X*]+\d{4}', r'\d{4}$'],
            group='payment',
        ),
        
        # Totals
        create_field(
            name='subtotal',
            field_type=FieldType.CURRENCY,
            labels=['Subtotal', 'Sub-Total'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='totals',
        ),
        create_field(
            name='tax',
            field_type=FieldType.CURRENCY,
            labels=['Tax', 'Sales Tax', 'VAT'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='totals',
        ),
        create_field(
            name='total',
            field_type=FieldType.CURRENCY,
            labels=['Total', 'Grand Total', 'Amount'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            required=True,
            group='totals',
        ),
        create_field(
            name='change',
            field_type=FieldType.CURRENCY,
            labels=['Change', 'Change Due'],
            patterns=[r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'],
            group='totals',
        ),
        
        # Items
        create_field(
            name='items',
            field_type=FieldType.TABLE,
            labels=['Item', 'Description', 'Product'],
            group='items',
        ),
    ],
    
    identification_patterns=[
        r'receipt',
        r'thank\s*you',
        r'transaction',
        r'change\s*due',
    ],
    identification_keywords=[
        'receipt', 'thank you', 'total', 'subtotal', 'tax',
        'cash', 'credit', 'change', 'cashier',
    ],
    
    table_fields=['items'],
    table_column_mappings={
        'items': [
            'item', 'description', 'product',
            'qty', 'quantity',
            'price', 'each',
            'amount', 'total',
        ],
    },
    
    category='financial',
    tags=['receipt', 'purchase', 'transaction'],
)


# =============================================================================
# RESUME TYPE
# =============================================================================

RESUME_TYPE = DocumentType(
    name='resume',
    display_name='Resume / CV',
    description='Professional resume or curriculum vitae',
    version='1.0',
    
    fields=[
        # Contact info
        create_field(
            name='name',
            field_type=FieldType.TEXT,
            labels=['Name'],
            position_hint='top',
            required=True,
            group='contact',
        ),
        create_field(
            name='email',
            field_type=FieldType.EMAIL,
            labels=['Email', 'E-mail'],
            patterns=[r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'],
            group='contact',
        ),
        create_field(
            name='phone',
            field_type=FieldType.PHONE,
            labels=['Phone', 'Tel', 'Mobile', 'Cell'],
            patterns=[
                r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            ],
            group='contact',
        ),
        create_field(
            name='address',
            field_type=FieldType.ADDRESS,
            labels=['Address', 'Location'],
            group='contact',
        ),
        create_field(
            name='linkedin',
            field_type=FieldType.TEXT,
            labels=['LinkedIn', 'linkedin.com'],
            patterns=[r'linkedin\.com/in/[\w-]+'],
            group='contact',
        ),
        create_field(
            name='website',
            field_type=FieldType.TEXT,
            labels=['Website', 'Portfolio', 'Blog'],
            patterns=[r'https?://[\w.-]+\.\w+'],
            group='contact',
        ),
        
        # Professional
        create_field(
            name='summary',
            field_type=FieldType.TEXT,
            labels=['Summary', 'Objective', 'Profile', 'About'],
            group='summary',
        ),
        create_field(
            name='experience',
            field_type=FieldType.LIST,
            labels=['Experience', 'Work Experience', 'Employment', 'Work History'],
            group='experience',
        ),
        create_field(
            name='education',
            field_type=FieldType.LIST,
            labels=['Education', 'Academic', 'Qualifications'],
            group='education',
        ),
        create_field(
            name='skills',
            field_type=FieldType.LIST,
            labels=['Skills', 'Technical Skills', 'Core Competencies', 'Expertise'],
            group='skills',
        ),
        create_field(
            name='certifications',
            field_type=FieldType.LIST,
            labels=['Certifications', 'Certificates', 'Licenses'],
            group='certifications',
        ),
        create_field(
            name='languages',
            field_type=FieldType.LIST,
            labels=['Languages', 'Language Skills'],
            group='languages',
        ),
    ],
    
    identification_patterns=[
        r'resume',
        r'curriculum\s*vitae',
        r'cv\b',
        r'experience',
        r'education',
    ],
    identification_keywords=[
        'resume', 'curriculum vitae', 'cv',
        'experience', 'education', 'skills',
        'objective', 'summary', 'employment',
    ],
    
    category='hr',
    tags=['resume', 'cv', 'employment', 'hr'],
)


# Register built-in types
def register_builtin_types():
    """Register all built-in document types."""
    from .registry import register_document_type
    
    for doc_type in [
        INVOICE_TYPE,
        BANK_STATEMENT_TYPE,
        RECEIPT_TYPE,
        RESUME_TYPE,
    ]:
        try:
            register_document_type(doc_type, overwrite=True)
        except Exception as e:
            pass  # Already registered
