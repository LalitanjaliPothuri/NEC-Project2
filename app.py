from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ASSETS_DIR,
    DEFAULT_COST_RATIO,
    DEFAULT_FORECAST_MONTHS,
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_SERVICE_LEVEL,
    DEFAULT_TEST_MONTHS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from src.data_processing import (
    build_monthly_summary,
    build_product_summary,
    generate_sample_data,
    infer_column_map,
    is_categorical_like,
    is_date_like,
    is_numeric_like,
    prepare_sales_dataframe,
    read_tabular_file,
    save_dataframe,
)
from src.inventory import build_inventory_report, stock_alert_summary
from src.modeling import load_model_bundle, train_sales_forecast_model


st.set_page_config(
    page_title="Intelligent Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def safe_filename(name: str) -> str:
    keep = [character if character.isalnum() or character in {".", "_", "-"} else "_" for character in name]
    return "".join(keep)


def currency(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"${value:,.2f}"


def compact_number(value: float) -> str:
    if pd.isna(value):
        return "-"
    magnitude = abs(float(value))
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">Intelligent Sales System</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[tuple[str, str, Optional[str]]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, delta) in zip(columns, metrics):
        with column:
            st.metric(label, value, delta)


def dashboard_stat_cards(metrics: list[tuple[str, str, Optional[str]]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, detail) in zip(columns, metrics):
        with column:
            detail_html = f"<div class='stat-detail'>{detail}</div>" if detail else ""
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                    {detail_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def download_csv_button(label: str, df: pd.DataFrame, file_name: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def set_mapping_defaults(df: pd.DataFrame) -> None:
    mapping = infer_column_map(df)
    st.session_state.column_map = mapping
    st.session_state.target_column = mapping.get("sales") or mapping.get("quantity") or mapping.get("revenue")
    st.session_state.revenue_column = mapping.get("revenue")
    st.session_state.price_column = mapping.get("price")
    st.session_state.cost_column = mapping.get("cost")
    st.session_state.stock_column = mapping.get("stock")


def role_candidates(df: pd.DataFrame, role: str) -> list[str]:
    columns = list(df.columns)
    detected = infer_column_map(df)
    suggested: list[str] = []

    if role == "date":
        preferred = [
            column
            for column in columns
            if is_date_like(df[column]) or "date" in column.lower() or "time" in column.lower()
        ]
    elif role == "product":
        preferred = [
            column
            for column in columns
            if is_categorical_like(df[column]) and not is_date_like(df[column])
            and any(token in column.lower() for token in ["product", "item", "sku", "name", "category", "brand"])
        ]
    elif role == "sales":
        preferred = [
            column
            for column in columns
            if is_numeric_like(df[column]) and any(token in column.lower() for token in ["sales", "qty", "quantity", "units", "demand", "volume"])
        ]
    elif role == "revenue":
        preferred = [
            column
            for column in columns
            if is_numeric_like(df[column]) and any(token in column.lower() for token in ["revenue", "amount", "turnover", "total", "net_sales"])
        ]
    elif role == "price":
        preferred = [
            column
            for column in columns
            if is_numeric_like(df[column]) and any(token in column.lower() for token in ["price", "rate", "mrp", "selling"])
        ]
    elif role == "cost":
        preferred = [
            column
            for column in columns
            if is_numeric_like(df[column]) and any(token in column.lower() for token in ["cost", "cogs", "purchase"])
        ]
    elif role == "stock":
        preferred = [
            column
            for column in columns
            if is_numeric_like(df[column]) and any(token in column.lower() for token in ["stock", "inventory", "on_hand", "available"])
        ]
    else:
        preferred = []

    hinted = detected.get(role)
    if hinted in columns:
        suggested.append(hinted)
    for column in preferred:
        if column not in suggested:
            suggested.append(column)

    if role == "date":
        fallback = [column for column in columns if is_date_like(df[column])]
    elif role == "product":
        fallback = [column for column in columns if is_categorical_like(df[column]) and not is_date_like(df[column])]
    elif role in {"sales", "revenue", "price", "cost", "stock"}:
        fallback = [column for column in columns if is_numeric_like(df[column])]
    else:
        fallback = []

    for column in fallback:
        if column not in suggested:
            suggested.append(column)
    return suggested


def list_local_datasets() -> list[Path]:
    files = sorted(
        [
            path
            for path in RAW_DATA_DIR.glob("*")
            if path.suffix.lower() in {".csv", ".xlsx", ".xls"}
        ],
        key=lambda path: path.name.lower(),
    )
    return files


def default_local_dataset() -> Path:
    demo_path = RAW_DATA_DIR / "demo_retail_dataset.csv"
    if demo_path.exists():
        return demo_path

    files = list_local_datasets()
    if files:
        return files[0]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    demo_df = generate_sample_data()
    save_dataframe(demo_df, demo_path)
    return demo_path


def load_dataset_from_path(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def reset_derived_state() -> None:
    st.session_state.clean_df = None
    st.session_state.monthly_df = None
    st.session_state.product_df = None
    st.session_state.model_bundle = None
    st.session_state.inventory_df = None


def load_demo_dataset() -> None:
    demo_path = RAW_DATA_DIR / "demo_retail_dataset.csv"
    if demo_path.exists():
        demo_df = load_dataset_from_path(demo_path)
    else:
        demo_df = generate_sample_data()
        save_dataframe(demo_df, demo_path)
    st.session_state.raw_df = demo_df
    st.session_state.raw_name = "demo_retail_dataset.csv"
    st.session_state.data_source = "Demo dataset"
    set_mapping_defaults(demo_df)
    reset_derived_state()
    process_current_data(auto_message=True)
    train_current_model(auto_message=True)


def process_current_data(auto_message: bool = False) -> None:
    raw_df = st.session_state.get("raw_df")
    if raw_df is None or raw_df.empty:
        st.warning("Upload a CSV or Excel file first.")
        return

    column_map = st.session_state.get("column_map", {})
    target_column = st.session_state.get("target_column")
    if not target_column:
        st.error("Select a sales or quantity column before preprocessing.")
        return

    date_column = column_map.get("date")
    product_column = column_map.get("product")
    if not date_column or date_column not in raw_df.columns or not is_date_like(raw_df[date_column]):
        st.error("Choose a valid date column that contains date values before preprocessing.")
        return
    if product_column and product_column in raw_df.columns and is_date_like(raw_df[product_column]):
        st.error("The product column cannot be date-like. Please choose a categorical product or item field.")
        return
    if target_column not in raw_df.columns:
        st.error("The selected sales target column is not available in the dataset.")
        return
    if not is_numeric_like(raw_df[target_column]):
        st.error("The selected sales target column must contain numeric demand or quantity values.")
        return

    for label, selected in {
        "Revenue": st.session_state.get("revenue_column"),
        "Unit price": st.session_state.get("price_column"),
        "Unit cost": st.session_state.get("cost_column"),
        "Stock": st.session_state.get("stock_column"),
    }.items():
        if selected and selected in raw_df.columns and not is_numeric_like(raw_df[selected]):
            st.error(f"The selected {label.lower()} column must contain numeric values.")
            return

    prepared = prepare_sales_dataframe(
        raw_df,
        column_map=column_map,
        target_column=target_column,
        revenue_column=st.session_state.get("revenue_column"),
        price_column=st.session_state.get("price_column"),
        cost_column=st.session_state.get("cost_column"),
        stock_column=st.session_state.get("stock_column"),
        default_cost_ratio=st.session_state.get("cost_ratio", DEFAULT_COST_RATIO),
    )

    monthly_df = build_monthly_summary(prepared)
    product_df = build_product_summary(prepared)

    save_dataframe(prepared, PROCESSED_DATA_DIR / "clean_sales.csv")
    save_dataframe(monthly_df, PROCESSED_DATA_DIR / "monthly_summary.csv")
    save_dataframe(product_df, PROCESSED_DATA_DIR / "product_summary.csv")

    st.session_state.clean_df = prepared
    st.session_state.monthly_df = monthly_df
    st.session_state.product_df = product_df
    st.session_state.inventory_df = None

    if auto_message:
        st.session_state.status_message = "Demo data prepared successfully."
    else:
        st.session_state.status_message = "Data preprocessing finished and saved to the processed folder."
        st.success(st.session_state.status_message)


def train_current_model(auto_message: bool = False) -> None:
    monthly_df = st.session_state.get("monthly_df")
    product_df = st.session_state.get("product_df")
    if monthly_df is None or monthly_df.empty:
        st.warning("Run preprocessing before training the model.")
        return

    bundle = train_sales_forecast_model(
        monthly_df,
        target_col="demand_units",
        forecast_months=st.session_state.get("forecast_months", DEFAULT_FORECAST_MONTHS),
        test_months=st.session_state.get("test_months", DEFAULT_TEST_MONTHS),
    )
    st.session_state.model_bundle = bundle
    st.session_state.forecast_df = bundle["forecast_frame"]
    st.session_state.validation_df = bundle["validation_frame"]
    inventory_df = build_inventory_report(
        product_df if product_df is not None else pd.DataFrame(),
        bundle["forecast_frame"],
        service_level=st.session_state.get("service_level", DEFAULT_SERVICE_LEVEL),
        lead_time_days=st.session_state.get("lead_time_days", DEFAULT_LEAD_TIME_DAYS),
    )
    st.session_state.inventory_df = inventory_df

    save_dataframe(bundle["forecast_frame"], REPORTS_DIR / "sales_forecast.csv")
    save_dataframe(bundle["validation_frame"], REPORTS_DIR / "validation_results.csv")
    save_dataframe(bundle["feature_importance"], REPORTS_DIR / "feature_importance.csv")
    if not inventory_df.empty:
        save_dataframe(inventory_df, REPORTS_DIR / "inventory_recommendations.csv")

    if auto_message:
        st.session_state.status_message = "Demo model trained successfully."
    else:
        st.session_state.status_message = "Model training finished and forecasts saved."
        st.success(st.session_state.status_message)


def load_or_bootstrap_state() -> None:
    if "forecast_months" not in st.session_state:
        st.session_state.forecast_months = DEFAULT_FORECAST_MONTHS
    if "test_months" not in st.session_state:
        st.session_state.test_months = DEFAULT_TEST_MONTHS
    if "lead_time_days" not in st.session_state:
        st.session_state.lead_time_days = DEFAULT_LEAD_TIME_DAYS
    if "service_level" not in st.session_state:
        st.session_state.service_level = DEFAULT_SERVICE_LEVEL
    if "cost_ratio" not in st.session_state:
        st.session_state.cost_ratio = DEFAULT_COST_RATIO

    if "raw_df" not in st.session_state:
        default_path = default_local_dataset()
        st.session_state.raw_df = load_dataset_from_path(default_path)
        st.session_state.raw_name = default_path.name
        st.session_state.data_source = "Demo dataset"
        set_mapping_defaults(st.session_state.raw_df)
        reset_derived_state()
        process_current_data(auto_message=True)
        train_current_model(auto_message=True)
        st.session_state.initialized = True
        return

    if st.session_state.get("clean_df") is None and st.session_state.get("raw_df") is not None:
        raw_df = st.session_state.raw_df
        if st.session_state.get("column_map") is None:
            set_mapping_defaults(raw_df)


def monthly_revenue_figure(monthly_df: pd.DataFrame, forecast_df: Optional[pd.DataFrame] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly_df["date"],
            y=monthly_df["revenue"],
            mode="lines+markers",
            name="Actual Revenue",
            line=dict(color="#f5b700", width=3),
        )
    )
    if forecast_df is not None and not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["predicted_revenue"],
                mode="lines+markers",
                name="Forecast Revenue",
                line=dict(color="#4cc9f0", width=3, dash="dash"),
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Month",
        yaxis_title="Revenue",
        template="plotly_dark",
        legend_title_text="",
    )
    return fig


def validation_figure(validation_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=validation_df["date"],
            y=validation_df["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#80ed99", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=validation_df["date"],
            y=validation_df["predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#ff7b00", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Month",
        yaxis_title="Sales",
        template="plotly_dark",
        legend_title_text="",
    )
    return fig


def top_products_figure(product_df: pd.DataFrame, value_col: str = "revenue", top_n: int = 10) -> go.Figure:
    frame = product_df.sort_values(value_col, ascending=False).head(top_n)
    fig = px.bar(
        frame,
        x=value_col,
        y="product",
        orientation="h",
        color=value_col,
        color_continuous_scale=["#1d3557", "#457b9d", "#a8dadc", "#f1faee"],
        title=f"Top {top_n} Products by {value_col.title()}",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10), template="plotly_dark", yaxis_title="")
    return fig


def profit_loss_figure(monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly_df["date"],
            y=monthly_df["profit"],
            name="Profit",
            marker_color="#6ee7b7",
        )
    )
    fig.add_trace(
        go.Bar(
            x=monthly_df["date"],
            y=-monthly_df["loss"],
            name="Loss",
            marker_color="#ef4444",
        )
    )
    fig.update_layout(
        barmode="relative",
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Month",
        yaxis_title="Profit / Loss",
        template="plotly_dark",
        legend_title_text="",
    )
    return fig


def inventory_figure(inventory_df: pd.DataFrame) -> go.Figure:
    frame = inventory_df.sort_values("recommended_order_qty", ascending=False).head(10)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["current_stock"],
            y=frame["product"],
            orientation="h",
            name="Current Stock",
            marker_color="#4cc9f0",
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["target_stock"],
            y=frame["product"],
            orientation="h",
            name="Target Stock",
            marker_color="#f5b700",
            opacity=0.55,
        )
    )
    fig.update_layout(
        barmode="overlay",
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Units",
        yaxis_title="Product",
        template="plotly_dark",
        legend_title_text="",
    )
    return fig


def dashboard_view() -> None:
    page_header(
        "Sales Forecast, Inventory, and Profit Intelligence",
        "Monitor sales, predict demand, estimate revenue, and keep stock levels aligned with future demand from one dashboard.",
    )

    clean_df = st.session_state.get("clean_df")
    monthly_df = st.session_state.get("monthly_df")
    product_df = st.session_state.get("product_df")
    model_bundle = st.session_state.get("model_bundle")
    inventory_df = st.session_state.get("inventory_df")

    if clean_df is None or monthly_df is None or product_df is None:
        st.info("Run preprocessing to unlock the full dashboard.")
        return

    total_revenue = float(clean_df["revenue"].sum())
    total_profit = float(clean_df["profit"].sum())
    annual_revenue = float(monthly_df.groupby("year")["revenue"].sum().max()) if not monthly_df.empty else 0.0
    top_product = product_df.iloc[0]["product"] if not product_df.empty else "-"
    top_product_revenue = float(product_df.iloc[0]["revenue"]) if not product_df.empty else 0.0
    forecast_revenue = float(model_bundle["forecast_frame"]["predicted_revenue"].sum()) if model_bundle else 0.0
    critical_count = int((inventory_df["status"] == "Critical").sum()) if inventory_df is not None and not inventory_df.empty else 0

    dashboard_stat_cards(
        [
            ("Total Revenue", currency(total_revenue), "All periods"),
            ("Estimated Annual Revenue", currency(annual_revenue), "Highest yearly total"),
            ("Gross Profit", currency(total_profit), "Revenue minus cost"),
            ("Top Product", top_product, currency(top_product_revenue)),
            ("Forecast Revenue", currency(forecast_revenue), "Future period total"),
            ("Critical Stock Items", str(critical_count), "Immediate restock risk"),
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(monthly_revenue_figure(monthly_df, model_bundle["forecast_frame"] if model_bundle else None), use_container_width=True)
    with col2:
        if model_bundle and not model_bundle["validation_frame"].empty:
            st.plotly_chart(validation_figure(model_bundle["validation_frame"]), use_container_width=True)
        else:
            st.subheader("Historical Revenue")
            st.line_chart(monthly_df.set_index("date")["revenue"])

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(top_products_figure(product_df, value_col="revenue", top_n=20), use_container_width=True)
    with col4:
        st.plotly_chart(profit_loss_figure(monthly_df), use_container_width=True)

    if inventory_df is not None and not inventory_df.empty:
        st.plotly_chart(inventory_figure(inventory_df), use_container_width=True)
        summary = stock_alert_summary(inventory_df)
        alert_cols = st.columns(4)
        labels = ["Critical", "Reorder Soon", "Overstock", "Healthy"]
        values = [summary["critical"], summary["reorder"], summary["overstock"], summary["healthy"]]
        for column, label, value in zip(alert_cols, labels, values):
            with column:
                st.metric(label, value)


def upload_view() -> None:
    page_header(
        "Data Upload",
        "Work directly from the datasets stored in the project folder, inspect the preview, and map business columns before preprocessing and forecasting.",
    )

    st.write("Place CSV or Excel files in the project data folder and pick them here. The app keeps the active dataset local, so no file browser is needed.")

    local_files = list_local_datasets()
    if not local_files:
        local_files = [default_local_dataset()]

    if len(local_files) == 1:
        selected_file = local_files[0]
        if st.session_state.get("raw_name") != selected_file.name:
            raw_df = load_dataset_from_path(selected_file)
            st.session_state.raw_df = raw_df
            st.session_state.raw_name = selected_file.name
            st.session_state.data_source = selected_file.name
            set_mapping_defaults(raw_df)
            reset_derived_state()
    else:
        options = {path.name: path for path in local_files}
        active_name = st.session_state.get("raw_name") if st.session_state.get("raw_name") in options else next(iter(options))
        selected_label = st.selectbox("Choose a local dataset", list(options.keys()), index=list(options.keys()).index(active_name))
        selected_file = options[selected_label]
        if st.button("Load selected dataset", use_container_width=True):
            raw_df = load_dataset_from_path(selected_file)
            st.session_state.raw_df = raw_df
            st.session_state.raw_name = selected_file.name
            st.session_state.data_source = selected_file.name
            set_mapping_defaults(raw_df)
            reset_derived_state()
            st.success(f"Loaded {selected_file.name} from the project data folder.")
            st.rerun()

    if st.button("Generate / refresh demo dataset", use_container_width=True):
        load_demo_dataset()
        st.rerun()

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("Select a local dataset or generate the demo dataset to continue.")
        return

    st.subheader("Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    detected = infer_column_map(raw_df)
    st.caption(f"Detected columns: {detected if detected else 'None'}")

    current_target = st.session_state.get("target_column")
    current_revenue = st.session_state.get("revenue_column")
    current_price = st.session_state.get("price_column")
    current_cost = st.session_state.get("cost_column")
    current_stock = st.session_state.get("stock_column")
    current_product = st.session_state.get("column_map", {}).get("product")
    current_date = st.session_state.get("column_map", {}).get("date")

    def choice_index(options: list[str], value: Optional[str]) -> int:
        return options.index(value) if value in options else 0

    with st.form("mapping_form"):
        left, right = st.columns(2)
        with left:
            date_options = ["-- None --"] + role_candidates(raw_df, "date")
            product_options = ["-- None --"] + role_candidates(raw_df, "product")
            target_options = ["-- None --"] + role_candidates(raw_df, "sales")
            date_col = st.selectbox("Date column", date_options, index=choice_index(date_options, current_date), key="date_mapping", help="Choose the transaction date used for monthly aggregation and forecasting.")
            product_col = st.selectbox("Product column", product_options, index=choice_index(product_options, current_product), key="product_mapping", help="Choose the item, SKU, or product name column.")
            target_col = st.selectbox("Sales / quantity column", target_options, index=choice_index(target_options, current_target), key="target_mapping", help="Choose the demand or units-sold field used as the forecasting target.")
        with right:
            revenue_options = ["-- None --"] + role_candidates(raw_df, "revenue")
            price_options = ["-- None --"] + role_candidates(raw_df, "price")
            cost_options = ["-- None --"] + role_candidates(raw_df, "cost")
            stock_options = ["-- None --"] + role_candidates(raw_df, "stock")
            revenue_col = st.selectbox("Revenue column", revenue_options, index=choice_index(revenue_options, current_revenue), key="revenue_mapping", help="Optional. Choose a direct revenue/amount field if your file already contains one.")
            price_col = st.selectbox("Unit price column", price_options, index=choice_index(price_options, current_price), key="price_mapping", help="Optional. Choose a unit selling price column if available.")
            cost_col = st.selectbox("Unit cost column", cost_options, index=choice_index(cost_options, current_cost), key="cost_mapping", help="Optional. Choose a unit cost / COGS column if available.")
            stock_col = st.selectbox("Stock column", stock_options, index=choice_index(stock_options, current_stock), key="stock_mapping", help="Optional. Choose the current stock / on-hand inventory column.")

        submitted = st.form_submit_button("Save mapping for preprocessing")

    if submitted:
        st.session_state.column_map = {
            "date": None if date_col == "-- None --" else date_col,
            "product": None if product_col == "-- None --" else product_col,
        }
        st.session_state.target_column = None if target_col == "-- None --" else target_col
        st.session_state.revenue_column = None if revenue_col == "-- None --" else revenue_col
        st.session_state.price_column = None if price_col == "-- None --" else price_col
        st.session_state.cost_column = None if cost_col == "-- None --" else cost_col
        st.session_state.stock_column = None if stock_col == "-- None --" else stock_col
        st.success("Mapping saved. Open Data Preprocessing to clean and aggregate the dataset.")


def preprocessing_view() -> None:
    page_header(
        "Data Preprocessing",
        "Clean dates, normalize sales fields, estimate revenue and profit, and build monthly summaries ready for forecasting.",
    )

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("Upload a dataset first.")
        return

    st.write("Current mapping")
    st.json(
        {
            "date": st.session_state.get("column_map", {}).get("date"),
            "product": st.session_state.get("column_map", {}).get("product"),
            "target": st.session_state.get("target_column"),
            "revenue": st.session_state.get("revenue_column"),
            "price": st.session_state.get("price_column"),
            "cost": st.session_state.get("cost_column"),
            "stock": st.session_state.get("stock_column"),
        }
    )

    raw_summary = pd.DataFrame(
        {
            "metric": ["Rows", "Columns", "Missing values", "Date range"],
            "value": [
                len(raw_df),
                raw_df.shape[1],
                int(raw_df.isna().sum().sum()),
                f"{pd.to_datetime(raw_df[st.session_state.get('column_map', {}).get('date')], errors='coerce').min()} to {pd.to_datetime(raw_df[st.session_state.get('column_map', {}).get('date')], errors='coerce').max()}"
                if st.session_state.get("column_map", {}).get("date") in raw_df.columns and is_date_like(raw_df[st.session_state.get("column_map", {}).get("date")])
                else "Unknown",
            ],
        }
    )
    st.dataframe(raw_summary, hide_index=True, use_container_width=True)

    if st.button("Run preprocessing now", use_container_width=True):
        process_current_data()
        st.rerun()

    clean_df = st.session_state.get("clean_df")
    monthly_df = st.session_state.get("monthly_df")
    product_df = st.session_state.get("product_df")
    if clean_df is None or monthly_df is None or product_df is None:
        st.warning("Preprocess the dataset to unlock the analysis pages.")
        return

    st.success("Cleaned dataset created successfully.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Clean rows", f"{len(clean_df):,}")
    with c2:
        st.metric("Monthly periods", f"{len(monthly_df):,}")
    with c3:
        st.metric("Products", f"{len(product_df):,}")

    st.subheader("Processed preview")
    st.dataframe(clean_df.head(20), use_container_width=True)
    st.subheader("Monthly summary")
    st.dataframe(monthly_df.head(12), use_container_width=True)


def eda_view() -> None:
    page_header(
        "EDA Analysis",
        "Explore the structure of the data, top products, sales seasonality, revenue patterns, and profit behavior.",
    )

    clean_df = st.session_state.get("clean_df")
    monthly_df = st.session_state.get("monthly_df")
    product_df = st.session_state.get("product_df")
    if clean_df is None or monthly_df is None or product_df is None:
        st.info("Run preprocessing before exploring the dataset.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(top_products_figure(product_df, value_col="demand_units", top_n=20), use_container_width=True)
    with col2:
        st.plotly_chart(monthly_revenue_figure(monthly_df), use_container_width=True)

    st.subheader("Product summary")
    st.dataframe(
        product_df.loc[:, ["product", "revenue", "demand_units", "profit", "current_stock"]]
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        numeric_cols = clean_df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            corr = clean_df[numeric_cols].corr().fillna(0)
            heatmap = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale="Viridis",
                )
            )
            heatmap.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(heatmap, use_container_width=True)
    with c2:
        monthly_trend = px.area(
            monthly_df,
            x="date",
            y="demand_units",
            title="Monthly Demand Trend",
        )
        monthly_trend.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(monthly_trend, use_container_width=True)

    st.subheader("Descriptive statistics")
    st.dataframe(clean_df.describe(include="all").transpose(), use_container_width=True)


def training_view() -> None:
    page_header(
        "Model Training",
        "Train the forecasting model, evaluate accuracy, and inspect which time features matter most.",
    )

    monthly_df = st.session_state.get("monthly_df")
    if monthly_df is None or monthly_df.empty:
        st.info("Preprocess the data before training the model.")
        return

    settings_cols = st.columns(3)
    with settings_cols[0]:
        st.metric("Forecast months", st.session_state.get("forecast_months", DEFAULT_FORECAST_MONTHS))
    with settings_cols[1]:
        st.metric("Validation months", st.session_state.get("test_months", DEFAULT_TEST_MONTHS))
    with settings_cols[2]:
        st.metric("Fallback cost ratio", f"{float(st.session_state.get('cost_ratio', DEFAULT_COST_RATIO)):.2f}")
    st.caption("Use the sidebar Pipeline settings to adjust these values.")

    if st.button("Train / retrain model", use_container_width=True):
        train_current_model()
        st.rerun()

    model_bundle = st.session_state.get("model_bundle")
    if model_bundle is None:
        st.info("Train the model to see validation metrics and forecast outputs.")
        return

    metrics = model_bundle["metrics"]
    metric_row(
        [
            ("MAE", f"{metrics['mae']:.2f}" if pd.notna(metrics["mae"]) else "-", None),
            ("RMSE", f"{metrics['rmse']:.2f}" if pd.notna(metrics["rmse"]) else "-", None),
            ("MAPE", f"{metrics['mape']:.2f}%" if pd.notna(metrics["mape"]) else "-", None),
            ("Feature Count", str(len(model_bundle["feature_cols"])), None),
        ]
    )

    if not model_bundle["validation_frame"].empty:
        st.plotly_chart(validation_figure(model_bundle["validation_frame"]), use_container_width=True)

    st.subheader("Feature importance")
    st.dataframe(model_bundle["feature_importance"], use_container_width=True)
    feature_chart = px.bar(model_bundle["feature_importance"].head(12), x="importance", y="feature", orientation="h", title="Top Model Features")
    feature_chart.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(feature_chart, use_container_width=True)


def forecasting_view() -> None:
    page_header(
        "Sales Forecasting",
        "See the next months of demand and revenue with historical context and model-generated forward projections.",
    )

    monthly_df = st.session_state.get("monthly_df")
    model_bundle = st.session_state.get("model_bundle")
    if monthly_df is None or monthly_df.empty:
        st.info("Preprocess the data first.")
        return

    if model_bundle is None:
        st.warning("Train the model to generate forecast charts.")
        return

    forecast_df = model_bundle["forecast_frame"]
    st.plotly_chart(monthly_revenue_figure(monthly_df, forecast_df), use_container_width=True)

    forecast_show = forecast_df.copy()
    forecast_show["predicted_units"] = forecast_show["predicted_units"].round(2)
    forecast_show["predicted_revenue"] = forecast_show["predicted_revenue"].round(2)
    st.dataframe(forecast_show, use_container_width=True)

    annual = monthly_df.groupby("year", as_index=False).agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
    annual_fig = go.Figure()
    annual_fig.add_trace(go.Bar(x=annual["year"].astype(str), y=annual["revenue"], name="Revenue", marker_color="#f5b700"))
    annual_fig.add_trace(go.Bar(x=annual["year"].astype(str), y=annual["profit"], name="Profit", marker_color="#80ed99"))
    annual_fig.update_layout(barmode="group", height=380, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    st.plotly_chart(annual_fig, use_container_width=True)


def inventory_view() -> None:
    page_header(
        "Inventory Optimization",
        "Identify stockouts, reorder points, and overstock risks using demand forecasts and service-level assumptions.",
    )

    product_df = st.session_state.get("product_df")
    model_bundle = st.session_state.get("model_bundle")
    if product_df is None or product_df.empty:
        st.info("Preprocess the data first.")
        return
    if model_bundle is None:
        st.warning("Train the forecasting model to produce inventory recommendations.")
        return

    if st.session_state.get("inventory_df") is None:
        st.session_state.inventory_df = build_inventory_report(
            product_df,
            model_bundle["forecast_frame"],
            service_level=st.session_state.get("service_level", DEFAULT_SERVICE_LEVEL),
            lead_time_days=st.session_state.get("lead_time_days", DEFAULT_LEAD_TIME_DAYS),
        )

    inventory_df = st.session_state.inventory_df
    if inventory_df is None or inventory_df.empty:
        st.info("Inventory recommendations are not available yet.")
        return

        settings_cols = st.columns(1)
        with settings_cols[0]:
            st.metric("Service level", f"{float(st.session_state.get('service_level', DEFAULT_SERVICE_LEVEL)):.2f}")
    st.caption("Adjust lead time and service level from the sidebar Pipeline settings.")

    if st.button("Refresh inventory recommendations", use_container_width=True):
        st.session_state.inventory_df = build_inventory_report(
            product_df,
            model_bundle["forecast_frame"],
            service_level=st.session_state.get("service_level", DEFAULT_SERVICE_LEVEL),
            lead_time_days=st.session_state.get("lead_time_days", DEFAULT_LEAD_TIME_DAYS),
        )
        inventory_df = st.session_state.inventory_df
        st.success("Inventory recommendations updated.")

    summary = stock_alert_summary(inventory_df)
    metric_row(
        [
            ("Critical", str(summary["critical"]), None),
            ("Reorder Soon", str(summary["reorder"]), None),
            ("Overstock", str(summary["overstock"]), None),
            ("Healthy", str(summary["healthy"]), None),
        ]
    )

    st.plotly_chart(inventory_figure(inventory_df), use_container_width=True)
    inventory_df = inventory_df.sort_values(["status", "days_until_stockout"], ascending=[True, True])
    st.dataframe(inventory_df, use_container_width=True)


def reports_view() -> None:
    page_header(
        "Reports",
        "Export the cleaned data, monthly summaries, forecasts, and inventory recommendations for business reporting.",
    )

    clean_df = st.session_state.get("clean_df")
    monthly_df = st.session_state.get("monthly_df")
    product_df = st.session_state.get("product_df")
    model_bundle = st.session_state.get("model_bundle")
    inventory_df = st.session_state.get("inventory_df")

    report_files = []
    if clean_df is not None:
        report_files.append(("Cleaned sales data", clean_df, "clean_sales.csv"))
    if monthly_df is not None:
        report_files.append(("Monthly summary", monthly_df, "monthly_summary.csv"))
    if product_df is not None:
        report_files.append(("Product summary", product_df, "product_summary.csv"))
    if model_bundle is not None:
        report_files.append(("Forecast", model_bundle["forecast_frame"], "sales_forecast.csv"))
        report_files.append(("Validation", model_bundle["validation_frame"], "validation_results.csv"))
        report_files.append(("Feature importance", model_bundle["feature_importance"], "feature_importance.csv"))
    if inventory_df is not None:
        report_files.append(("Inventory recommendations", inventory_df, "inventory_recommendations.csv"))

    if not report_files:
        st.info("No reports are ready yet. Run preprocessing and model training first.")
        return

    for title, frame, filename in report_files:
        st.subheader(title)
        st.dataframe(frame.head(20), use_container_width=True)
        download_csv_button(f"Download {title}", frame, filename)
        st.divider()


def sidebar_controls() -> str:
    st.sidebar.title("Intelligent Sales")
    st.sidebar.caption("Upload data, train the model, and inspect forecast and inventory intelligence.")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Data Upload",
            "Data Preprocessing",
            "EDA Analysis",
            "Model Training",
            "Sales Forecasting",
            "Inventory Optimization",
            "Reports",
        ],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Pipeline settings")
    st.sidebar.number_input("Forecast months", min_value=3, max_value=24, value=st.session_state.get("forecast_months", DEFAULT_FORECAST_MONTHS), key="forecast_months")
    st.sidebar.number_input("Validation months", min_value=2, max_value=12, value=st.session_state.get("test_months", DEFAULT_TEST_MONTHS), key="test_months")
    st.sidebar.slider("Lead time days", min_value=7, max_value=60, value=st.session_state.get("lead_time_days", DEFAULT_LEAD_TIME_DAYS), key="lead_time_days")
    st.sidebar.slider("Service level", min_value=0.80, max_value=0.99, value=float(st.session_state.get("service_level", DEFAULT_SERVICE_LEVEL)), step=0.01, key="service_level")
    st.sidebar.slider("Fallback cost ratio", min_value=0.30, max_value=0.90, value=float(st.session_state.get("cost_ratio", DEFAULT_COST_RATIO)), step=0.01, key="cost_ratio")

    st.sidebar.divider()
    if st.sidebar.button("Reload demo dataset", use_container_width=True):
        load_demo_dataset()
        st.rerun()

    if st.session_state.get("raw_name"):
        st.sidebar.success(f"Active dataset: {st.session_state.get('raw_name')}")
    return page


def main() -> None:
    load_css()
    load_or_bootstrap_state()
    page = sidebar_controls()

    if page == "Dashboard":
        dashboard_view()
    elif page == "Data Upload":
        upload_view()
    elif page == "Data Preprocessing":
        preprocessing_view()
    elif page == "EDA Analysis":
        eda_view()
    elif page == "Model Training":
        training_view()
    elif page == "Sales Forecasting":
        forecasting_view()
    elif page == "Inventory Optimization":
        inventory_view()
    elif page == "Reports":
        reports_view()


if __name__ == "__main__":
    main()
