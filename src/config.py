from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

DEFAULT_FORECAST_MONTHS = 12
DEFAULT_TEST_MONTHS = 6
DEFAULT_LEAD_TIME_DAYS = 30
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_COST_RATIO = 0.65
RANDOM_STATE = 42

COLUMN_ALIASES = {
    "date": ["date", "order_date", "invoice_date", "transaction_date", "day", "timestamp"],
    "product": ["product", "item", "sku", "product_name", "item_name", "category", "brand"],
    "sales": ["sales", "units", "units_sold", "quantity", "qty", "demand", "volume"],
    "revenue": ["revenue", "sales_amount", "amount", "turnover", "total_sales", "net_sales"],
    "price": ["price", "unit_price", "sale_price", "selling_price", "mrp"],
    "cost": ["cost", "unit_cost", "purchase_cost", "cogs", "cost_price"],
    "stock": ["stock", "inventory", "on_hand", "available_stock", "current_stock", "qty_in_stock"],
    "region": ["region", "state", "city", "market", "territory"],
}
