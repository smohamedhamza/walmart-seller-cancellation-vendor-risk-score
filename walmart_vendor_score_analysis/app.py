
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer

# ------------------------------------------------------------------------------
# 1. Config & Caching
# ------------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Walmart Seller Cancellation Analysis")

@st.cache_data
# ------------------------------------------------------------------------------
# 2. Helper Functions (Copied/Adapted from Original Script)
# ------------------------------------------------------------------------------

def compute_2x2(a, b, c, d):
    """Return full set of disproportionality metrics"""
    # Avoid division errors
    if any(x == 0 for x in [a,b,c,d]):
        return {
            "OR": np.nan, "OR_low": np.nan, "OR_high": np.nan,
            "RR": np.nan, "RR_low": np.nan, "RR_high": np.nan,
            "chi2": np.nan,
            "IC": np.nan, "IC025": np.nan,
            "EBGM": np.nan, "EBGM05": np.nan
        }

    # --- OR ---
    OR = (a/b) / (c/d)
    se_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo_or = np.exp(np.log(OR) - 1.96 * se_or)
    hi_or = np.exp(np.log(OR) + 1.96 * se_or)

    # --- RR ---
    RR = (a / (a+b)) / (c / (c+d))
    se_rr = np.sqrt((1/a) - (1/(a+b)) + (1/c) - (1/(c+d)))
    lo_rr = np.exp(np.log(RR) - 1.96 * se_rr)
    hi_rr = np.exp(np.log(RR) + 1.96 * se_rr)

    # --- Chi-square ---
    chi2 = ((a*d - b*c)**2 * (a+b+c+d)) / ((a+b)*(c+d)*(a+c)*(b+d))

    # --- IC (log2 disproportionality) ---
    IC = np.log2((a * (a+b+c+d)) / ((a+b)*(a+c)))
    IC025 = IC - 1.96 * np.sqrt(1/a + 1/b + 1/c + 1/d)

    # --- EBGM ---
    EBGM = (a * (a+b+c+d)) / ((a+b)*(a+c))
    EBGM05 = EBGM - 1.96 * np.sqrt(1/a + 1/b + 1/c + 1/d)

    return {
        "OR": OR, "OR_low": lo_or, "OR_high": hi_or,
        "RR": RR, "RR_low": lo_rr, "RR_high": hi_rr,
        "chi2": chi2,
        "IC": IC, "IC025": IC025,
        "EBGM": EBGM, "EBGM05": EBGM05
    }

def risk_flags(metrics):
    """Return multi-level risk categories + overall signal"""
    # OR Risk Category
    if not np.isnan(metrics["OR_low"]) and metrics["OR_low"] > 1:
        OR_label = "High Risk"
        OR_flag = 1
    elif not np.isnan(metrics["OR_high"]) and metrics["OR_high"] < 1:
        OR_label = "Low Risk"
        OR_flag = -1
    else:
        OR_label = "Insignificant"
        OR_flag = 0

    # RR Risk Category
    if not np.isnan(metrics["RR_low"]) and metrics["RR_low"] > 1:
        RR_label = "High Risk"
        RR_flag = 1
    elif not np.isnan(metrics["RR_high"]) and metrics["RR_high"] < 1:
        RR_label = "Low Risk"
        RR_flag = -1
    else:
        RR_label = "Insignificant"
        RR_flag = 0

    # OVERALL SIGNAL
    overall_sig = int(
        (metrics["OR_low"] > 1 if not np.isnan(metrics["OR_low"]) else False) and
        (metrics["RR"] > 2 if not np.isnan(metrics["RR"]) else False) and
        (metrics["IC025"] > 0 if not np.isnan(metrics["IC025"]) else False) and
        (metrics["EBGM05"] > 2 if not np.isnan(metrics["EBGM05"]) else False) and
        (metrics["chi2"] >= 4 if not np.isnan(metrics["chi2"]) else False)
    )

    return {
        "risk_indicator": overall_sig,
        "OR_risk_flag": OR_flag,
        "RR_risk_flag": RR_flag,
        "OR_risk_label": OR_label,
        "RR_risk_label": RR_label
    }

def build_monthly_dispro_table(df, group_col):
    """
    Builds the monthly disproportionality table. 
    Note: For the UI, we might be running this on a filtered dataset, 
    but the logic remains the same.
    """
    df = df.copy()
    # Ensure datetime already handled before calling this, but safe to redo
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["month"] = df["order_date"].dt.to_period("M").astype(str)

    groups = df[group_col].unique()
    months = sorted(df["month"].unique())

    full_results = []
    
    # Using a simple loop instead of tqdm for Streamlit (could use st.progress but might be overkill)
    for grp in groups:
        grp_df = df[df[group_col] == grp]

        # -------- OVERALL --------
        a = grp_df["is_seller_cancel"].sum()
        b = len(grp_df) - a

        other = df[df[group_col] != grp]
        c = other["is_seller_cancel"].sum()
        d = len(other) - c

        metrics = compute_2x2(a,b,c,d)
        flags = risk_flags(metrics)

        row = {
            group_col: grp,
            "overall_total_orders": len(grp_df),
            "overall_cancelled_orders": a,
            "overall_SCP": round(a/len(grp_df)*100,2) if len(grp_df)>0 else 0,
            "overall_SCC": round(a/df["is_seller_cancel"].sum()*100,2) if df["is_seller_cancel"].sum()>0 else 0,
            **{f"overall_{k}": v for k,v in metrics.items()},
            **{f"overall_{k}": v for k,v in flags.items()}
        }

        # -------- MONTHLY --------
        for m in months:
            m_df = df[df["month"] == m]
            grp_mdf = m_df[m_df[group_col] == grp]
            other_mdf = m_df[m_df[group_col] != grp]

            a = grp_mdf["is_seller_cancel"].sum()
            b = len(grp_mdf) - a
            c = other_mdf["is_seller_cancel"].sum()
            d = len(other_mdf) - c

            metrics = compute_2x2(a,b,c,d)
            flags = risk_flags(metrics)

            pref = m.replace("-", "_")

            row.update({
                f"y_m_{pref}_total_orders": len(grp_mdf),
                f"y_m_{pref}_cancelled_orders": a,
                f"y_m_{pref}_SCP": round(a/len(grp_mdf)*100,2) if len(grp_mdf)>0 else 0,
                f"y_m_{pref}_SCC": round(a/m_df["is_seller_cancel"].sum()*100,2) if m_df["is_seller_cancel"].sum()>0 else 0,
                **{f"{pref}_{k}": v for k,v in metrics.items()},
                **{f"{pref}_{k}": v for k,v in flags.items()}
            })

        full_results.append(row)

    full_results = pd.DataFrame(full_results)

    # ---------- Fill + Round ----------
    numeric_keywords = ["OR", "RR", "chi2", "IC", "EBGM"]
    metric_cols = [
        c for c in full_results.columns
        if any(k.lower() in c.lower() for k in numeric_keywords)
    ]
    if not full_results.empty:
        full_results[metric_cols] = full_results[metric_cols].fillna(0)
        full_results[metric_cols] = full_results[metric_cols].round(3)

    return full_results

# ------------------------------------------------------------------------------
# 3. Main Application Logic
# ------------------------------------------------------------------------------

def main():
    st.title("Walmart Seller Cancellation & Risk Analysis")

    # --- Load Data ---
    with st.spinner("Loading data..."):
        #original_df, _, product_cat_df = load_data()
        import pandas as pd
        #original_df = pd.read_parquet("C:/Users/mohamedh/Downloads/OJC projects/2025/10 October/Walmart refund analysis/cancellation results/new results 26-12/seller_cancellation_merged_1.parquet")
        original_df = pd.read_parquet("walmart_vendor_score_analysis/seller_cancellation_merged_1.parquet")
    
    
    # Ensure ordered date
    import pandas as pd
    original_df["order_date"] = pd.to_datetime(original_df["order_date"], errors="coerce")

    # --- Sidebar: Filter ---
    st.sidebar.header("Filters")
    
    min_date = original_df["order_date"].min().date()
    max_date = original_df["order_date"].max().date()
    
    # Default start as in script
    default_start = pd.to_datetime("2022-01-01").date()
    default_end = pd.to_datetime("2022-02-01").date()
    if default_start < min_date: default_start = min_date
    
    start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=default_end, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    # Filter Data
    # Convert to timestamps for comparison
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)
    
    filtered_df_base = original_df[
        (original_df["order_date"] >= ts_start) & 
        (original_df["order_date"] <= ts_end)
    ].copy()

    if filtered_df_base.empty:
        st.warning("No data found for the selected date range.")
        return

    st.write(f"Displaying data from **{start_date}** to **{end_date}** ({len(filtered_df_base)} records)")

    # --------------------------------------------------------------------------
    # 4. Small Table: Cancellation Counts
    # --------------------------------------------------------------------------
    st.subheader("Cancellation Initiated Counts Summary")
    
    # Group by cancellation_initiated and cancellation_reason_clean
    cancel_counts = filtered_df_base.groupby(
        ["cancellation_initiated"] #, "cancellation_reason_clean"]
    )["order_id"].nunique().reset_index(name="Count")
    
    cancel_counts = cancel_counts.sort_values(by="Count", ascending=False)
    st.dataframe(cancel_counts, use_container_width=True)

    total_orders = filtered_df_base["order_id"].nunique()
    seller_cancel_orders = (
        filtered_df_base[
            filtered_df_base["cancellation_initiated"].isin(["Internal Cancellation", "Vendor Initiated"])
        ]["order_id"]
        .nunique()
        )
    
    customer_cancel_orders = (
        filtered_df_base[
            filtered_df_base["cancellation_initiated"] == "Customer Request"
        ]["order_id"]
        .nunique()
    )

    safe_orders = (
        filtered_df_base[
            filtered_df_base["cancellation_initiated"] == "Not cancelled"
        ]["order_id"]
        .nunique()
    )

    seller_cancel_pct   = (seller_cancel_orders / total_orders) * 100
    customer_cancel_pct = (customer_cancel_orders / total_orders) * 100
    safe_orders_pct     = (safe_orders / total_orders) * 100


    st.markdown("### Cancellation Mix (%)")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Seller Cancellation %",
        value=f"{seller_cancel_pct:.2f}%",
        delta=f"{seller_cancel_orders} orders",
        help="Internal + Vendor initiated cancellations / Total orders"
    )

    col2.metric(
        label="Customer Cancellation %",
        value=f"{customer_cancel_pct:.2f}%",
        delta=f"{customer_cancel_orders} orders"
    )

    col3.metric(
        label="Safe Orders %",
        value=f"{safe_orders_pct:.2f}%",
        delta=f"{safe_orders} orders"        
    )

    st.subheader("Cancellation Reasons Counts Summary")
    
    # Group by cancellation_initiated and cancellation_reason_clean
    cancel_reason_counts = filtered_df_base.groupby(
        ["cancellation_reason_clean"]#, "cancellation_reason_clean"]
    )["order_id"].nunique().reset_index(name="Count")
    
    cancel_reason_counts = cancel_reason_counts.sort_values(by="Count", ascending=False)
    st.dataframe(cancel_reason_counts, use_container_width=True)

    # --------------------------------------------------------------------------
    # 5. Full Pipeline Processing (Vendor Analysis)
    # --------------------------------------------------------------------------
    # We need to replicate the pipeline for the filtered data to get dynamic results
    
    with st.spinner("Running analysis pipeline... This may take a moment."):

        seller_cancellation_merged_1_vendor_analysis = filtered_df_base[["order_id","order_date", "vendor_id", "vendor_name","cancellation_initiated"]]

        seller_cancellation_merged_1_vendor_analysis = (
            seller_cancellation_merged_1_vendor_analysis
            .drop_duplicates(subset=["order_id","vendor_id"], keep="last")
        )
        
        df = seller_cancellation_merged_1_vendor_analysis.copy()

        seller_cancel_values = ["Vendor Initiated", "OJC", "Internal Cancellation"]
        df["is_seller_cancel"] = df["cancellation_initiated"].isin(seller_cancel_values).astype(int)

        vendor_monthly_risk = build_monthly_dispro_table(df,group_col = "vendor_name")

        vendor_monthly_risk = vendor_monthly_risk[
            vendor_monthly_risk["vendor_name"].astype(str).str.strip() != "0"]

        mask = filtered_df_base["confirmed_shelf"].isna()
        mask_1 = filtered_df_base["confirmed_category"].isna()

        filtered_df_base.loc[mask_1, "confirmed_category"] = "Unknown"
        filtered_df_base.loc[mask, "confirmed_shelf"] = "Unknown"

        seller_cancellation_merged_1_conf_cat_analysis = filtered_df_base[["order_id","order_date", "confirmed_category","cancellation_initiated"]]

        seller_cancellation_merged_1_conf_cat_analysis = (
            seller_cancellation_merged_1_conf_cat_analysis
            .drop_duplicates(subset=["order_id","confirmed_category"], keep="last")
        )

        df2 = seller_cancellation_merged_1_conf_cat_analysis.copy()

        seller_cancel_values = ["Vendor Initiated", "OJC", "Internal Cancellation"]
        df2["is_seller_cancel"] = df2["cancellation_initiated"].isin(seller_cancel_values).astype(int)

        category_monthly_risk = build_monthly_dispro_table(df2,group_col = "confirmed_category")

        seller_cancellation_merged_1_product_wise_analysis = filtered_df_base[["order_id","order_date", "pid","product_name","cancellation_initiated"]]

        seller_cancellation_merged_1_product_wise_analysis = (
            seller_cancellation_merged_1_product_wise_analysis
            .drop_duplicates(subset=["order_id","pid"], keep="last")
        )

        df_3 = seller_cancellation_merged_1_product_wise_analysis.copy()

        seller_cancel_values = ["Vendor Initiated", "OJC", "Internal Cancellation"]
        df_3["is_seller_cancel"] = df_3["cancellation_initiated"].isin(seller_cancel_values).astype(int)        

        final_df = filtered_df_base.copy()

        final_df["vendor_id"] = final_df["vendor_id"].astype(str)
        final_df["vendor_name"] = final_df["vendor_name"].astype(str)
        final_df["confirmed_category"] = final_df["confirmed_category"].astype(str)
        vendor_monthly_risk["vendor_name"] = vendor_monthly_risk["vendor_name"].astype(str)
        category_monthly_risk["confirmed_category"] = category_monthly_risk["confirmed_category"].astype(str)


        vendor_monthly_risk_merge = vendor_monthly_risk.copy()

        vendor_monthly_risk_merge_subset = vendor_monthly_risk_merge[["vendor_name","overall_SCP","overall_SCC","overall_OR","overall_RR",
        "overall_risk_indicator","overall_OR_risk_flag","overall_RR_risk_flag"]]

        vendor_monthly_risk_merge_subset_cols = [c for c in vendor_monthly_risk_merge_subset.columns if c not in ["vendor_name"]]
        vendor_monthly_risk_merge_subset = vendor_monthly_risk_merge_subset.rename(
            columns={col: f"{col}_vendor" for col in vendor_monthly_risk_merge_subset_cols}
        )

        final_df = final_df.merge(
            vendor_monthly_risk_merge_subset,
            on="vendor_name",
            how="left"
        )


        category_monthly_risk_merge = category_monthly_risk.copy()

        category_monthly_risk_merge_subset = category_monthly_risk_merge[["confirmed_category","overall_SCP","overall_SCC","overall_OR","overall_RR",
        "overall_risk_indicator","overall_OR_risk_flag","overall_RR_risk_flag"]]

        category_monthly_risk_merge_subset_cols = [c for c in category_monthly_risk_merge_subset.columns if c not in ["confirmed_category"]]
        category_monthly_risk_merge_subset = category_monthly_risk_merge_subset.rename(
            columns={col: f"{col}_conf_cat" for col in category_monthly_risk_merge_subset_cols}
        )

        final_df = final_df.merge(
            category_monthly_risk_merge_subset,
            on="confirmed_category",
            how="left"
        )

        # Ensure order_date is datetime
        final_df["order_date"] = pd.to_datetime(final_df["order_date"], errors="coerce")


        # Extract month name (Jan, Feb, ...)
        final_df["order_month"] = final_df["order_date"].dt.strftime("%b")


        # Extract day of week name (Monday, Tuesday, ...)
        final_df["order_day_of_week"] = final_df["order_date"].dt.strftime("%A")

        final_df = final_df.dropna(subset=["itemid"])

        import numpy as np
        import pandas as pd

        # ---------------------------
        # 1️⃣ Lead Time Comparison
        # ---------------------------
        def compare_leadtime(row):
            lt = row["leadtime_days"]
            olt = row["original_leadtime_days"]

            if pd.isna(lt) or pd.isna(olt):
                return "Missing"
            elif lt > olt:
                return "Delayed"
            else:
                return "On Time"

        final_df["leadtime_status"] = final_df.apply(compare_leadtime, axis=1)

        # ---------------------------
        # 2️⃣ Posted Price vs Price
        # ---------------------------
        def compare_price(row):
            posted = row["posted_price"]
            actual = row["price"]

            if pd.isna(posted) or pd.isna(actual):
                return "Missing"
            elif posted > actual:
                return "Posted Higher"
            elif posted < actual:
                return "Posted Lower"
            else:
                return "Same"

        final_df["price_relation"] = final_df.apply(compare_price, axis=1)

        # ---------------------------
        # 3️⃣ Quantity vs Availability (status_at_time_of_order_qty)
        # ---------------------------
        def compare_qty(row):
            order_qty = row["qty"]
            available_qty = row["status_at_time_of_order_qty"]

            if pd.isna(order_qty) or pd.isna(available_qty):
                return "Unknown"
            elif available_qty >= order_qty:
                return "Sufficient"
            else:
                return "Insufficient"

        final_df["can_fulfill_order"] = final_df.apply(compare_qty, axis=1)

        cols_to_fill = ["usps_invalid_status", "rdi", "iv_status"]
        
        final_df[cols_to_fill] = final_df[cols_to_fill].fillna("Unknown")

        final_df["iv_status"] = final_df["iv_status"].replace(["", " "], np.nan)
        final_df["iv_status"] = final_df["iv_status"].fillna("Unknown")

        # Ensure date columns are proper datetime
        final_df["order_date"] = pd.to_datetime(final_df["order_date"], errors="coerce")
        final_df["promised_ship_date"] = pd.to_datetime(final_df["promised_ship_date"], errors="coerce")
        final_df["promised_delivery_date"] = pd.to_datetime(final_df["promised_delivery_date"], errors="coerce")

        # ---- Difference in days ----
        final_df["ship_order_days"] = (final_df["order_date"] - final_df["promised_ship_date"]).dt.days
        final_df["delivery_order_days"] = (final_df["order_date"] - final_df["promised_delivery_date"]).dt.days

        confcat_cols = [col for col in final_df.columns if col.endswith("_conf_cat")]
        
        final_df[confcat_cols] = final_df[confcat_cols].fillna(0)

        product_categorization_fragility = pd.read_parquet("walmart_vendor_score_analysis/product_categorization_nlm.parquet")
        product_categorization_fragility_3 = pd.read_parquet("walmart_vendor_score_analysis/product_categorization_nlm_rest.parquet")

        product_categorization_fragility_final = pd.concat(
            [product_categorization_fragility, product_categorization_fragility_3],
            ignore_index=True
        )

        product_categorization_fragility_final = (
            product_categorization_fragility_final
            .drop_duplicates(subset="product_id", keep="last")
        )

        product_categorization_fragility_final_subset = (
            product_categorization_fragility_final[["product_id", "fragility"]].copy()
        )

        # Convert to string for safe merging
        final_df["pid"] = final_df["pid"].astype(str)
        product_categorization_fragility_final_subset["product_id"] = \
            product_categorization_fragility_final_subset["product_id"].astype(str)

        # Now merge safely
        final_df = final_df.merge(
            product_categorization_fragility_final_subset,
            left_on="pid",
            right_on="product_id",
            how="left"
        )

        # Remove duplicate product_id column from merged data
        final_df = final_df.drop(columns=["product_id"])

        # Fill missing fragility
        final_df["fragility"] = final_df["fragility"].fillna("Unknown")

        final_df["status_at_time_of_order_cat_updated"] = np.where(
            final_df["status_at_time_of_order_cat"] == "Available",
            "Available",
            "Unavailable"
        )

        import numpy as np
        import pandas as pd

        # Ensure datetime
        final_df["status_date"] = pd.to_datetime(final_df["status_date"], errors="coerce")
        final_df["promised_ship_date"] = pd.to_datetime(final_df["promised_ship_date"], errors="coerce")

        # shipped_ind
        final_df["shipped_ind"] = (final_df["status"] == "SHIPPED").astype(int)

        # actual_ship_date
        final_df["actual_ship_date"] = np.where(
            final_df["shipped_ind"] == 1,
            final_df["status_date"],
            pd.NaT
        )

        final_df["actual_ship_date"] = pd.to_datetime(final_df["actual_ship_date"], errors="coerce")

        # Ensure datetime
        final_df["srvc_ship_status_date"] = pd.to_datetime(
            final_df["srvc_ship_status_date"], errors="coerce"
        )

        # delivered_ind (case-insensitive)
        final_df["delivered_ind"] = (
            final_df["srvc_ship_status"]
            .str.upper()
            .eq("DELIVERED")
            .fillna(False)
            .astype(int)
        )

        # actual_delivery_date
        final_df["actual_delivery_date"] = np.where(
            final_df["delivered_ind"] == 1,
            final_df["srvc_ship_status_date"],
            pd.NaT
        )

        final_df["actual_delivery_date"] = pd.to_datetime(final_df["actual_delivery_date"], errors="coerce")

        final_df["ship_order_days_actual"] = (final_df["actual_ship_date"] - final_df["order_date"]).dt.days
        final_df["delivery_order_days_actual"] = (final_df["actual_delivery_date"] - final_df["order_date"]).dt.days

        import pandas as pd

        def derive_timing_status(
            actual_date,
            promised_date,
            final_cancellation_date_1,
            today=None
        ):
            if today is None:
                today = pd.Timestamp.today().normalize()

            # ---------------------------
            # 1️⃣ CANCELLED CASES
            # ---------------------------
            if pd.notna(final_cancellation_date_1):

                # Promised missing
                if pd.isna(promised_date):
                    return "Cancelled – Promised Missing"

                # Actual missing → compare cancellation vs promised
                if pd.isna(actual_date):
                    if final_cancellation_date_1 < promised_date:
                        return "Cancelled Before Promised"
                    else:
                        return "Cancelled After Promised"

                # Both actual & promised present
                if actual_date < promised_date:
                    return "Early – Cancelled"
                elif actual_date == promised_date:
                    return "On-time – Cancelled"
                else:
                    return "Late – Cancelled"

            # ---------------------------
            # 2️⃣ NOT CANCELLED (YOUR ORIGINAL LOGIC)
            # ---------------------------

            # Both missing
            if pd.isna(actual_date) and pd.isna(promised_date):
                return "Both Missing"

            # Promised missing
            if pd.isna(promised_date):
                return "Promised Missing"

            # Actual missing → check against promised
            if pd.isna(actual_date):
                if promised_date < today:
                    return "Actual Missing Past Promised"
                else:
                    return "Actual Missing Within Promised"

            # Both present
            if actual_date < promised_date:
                return "Early"
            elif actual_date == promised_date:
                return "On-time"
            else:
                return "Late"



        final_df["ship_status"] = final_df.apply(
            lambda r: derive_timing_status(
                r["actual_ship_date"],
                r["promised_ship_date"],
                r["final_cancellation_date_1"]
            ),
            axis=1
        )


        final_df["promised_delivery_date"] = pd.to_datetime(
            final_df["promised_delivery_date"], errors="coerce"
        )

        final_df["delivery_status"] = final_df.apply(
            lambda r: derive_timing_status(
                r["actual_delivery_date"],
                r["promised_delivery_date"],
                r["final_cancellation_date_1"]
            ),
            axis=1
        )

        final_df["cancellation_order_days"] = (final_df["final_cancellation_date_1"] - final_df["order_date"]).dt.days
        final_df["cancellation_ship_days"] = (final_df["promised_ship_date"] - final_df["final_cancellation_date_1"]).dt.days
        final_df["cancellation_delivery_days"] = (final_df["promised_delivery_date"] - final_df["final_cancellation_date_1"]).dt.days        

        final_df_vendor_analysis = final_df[["order_id","order_date", "vendor_id", "vendor_name","cancellation_initiated","cancellation_reason_clean",
        "cancellation_reason_desc","order_month","order_day_of_week","leadtime_status","price_relation","can_fulfill_order","fragility","status_at_time_of_order_cat",
        "status_at_time_of_order_qty","status_at_time_of_order_cat_updated","confirmed_category","product_name","pid","itemid", "state","city","zip","availability",
        "vendor_disabled","naomi_vendor","status","status_date", "srvc_ship_method","srvc_ship_status","srvc_ship_status_date","promised_ship_date","promised_delivery_date",
        "actual_ship_date","actual_delivery_date","final_cancellation_date_1","ship_order_days","delivery_order_days","cancellation_order_days","cancellation_ship_days","cancellation_delivery_days",
        "ship_order_days_actual","delivery_order_days_actual",
        "isLTL","shipped_ind","delivered_ind","ship_status","delivery_status"]]

        #final_df_vendor_analysis = final_df_vendor_analysis[final_df_vendor_analysis["order_date"] >= '2025-01-01']

        final_df_vendor_analysis = (
            final_df_vendor_analysis
            .drop_duplicates(subset=["order_id","vendor_id"], keep="last")
        )

        SELLER_CANCEL = ["Vendor Initiated", "OJC", "Internal Cancellation"]
        CUSTOMER_CANCEL = ["Customer Request"]

        REASON_MAP = {
            "Expected Late": ["Expected Late"],
            "Back Order": ["Backorder", "Extended Backorder",],
            "Discontinued": ["Discontinued"],
            "Pricing Issue": ["Pricing"],
            "Listing Error": ["Listing Error"],
            "OJ Requested": ["OJ Requested"],
            "High Shipping Cost": ["High Shipping Cost"]
        }

        SAFE_STATUSES = ["Not cancelled"]

        df = final_df_vendor_analysis.copy()

        df["is_seller_cancel"] = df["cancellation_initiated"].isin(SELLER_CANCEL)
        df["is_customer_cancel"] = df["cancellation_initiated"].isin(CUSTOMER_CANCEL)
        df["is_not_cancelled"] = df["cancellation_initiated"].isin(SAFE_STATUSES)

        # Normalize reason text
        df["reason_lc"] = df["cancellation_reason_clean"].str.lower().fillna("")

        vendor_summary = (
            df.groupby("vendor_name")
            .agg(
                # -------------------
                # Orders
                # -------------------
                total_orders=("order_id", "nunique"),
                seller_cancel_orders=("is_seller_cancel", "sum"),
                customer_cancel_orders=("is_customer_cancel", "sum"),
                safe_orders=("is_not_cancelled", "sum"),

                # -------------------
                # Cancellation reasons
                # -------------------
                expected_late_orders=("reason_lc", lambda x: x.str.contains("expected late").sum()),
                backorder_orders=("reason_lc", lambda x: x.str.contains("backorder|extended backorder").sum()),
                discontinued_orders=("reason_lc", lambda x: x.str.contains("discontinued").sum()),
                pricing_issue_orders=("reason_lc", lambda x: x.str.contains("pricing").sum()),
                listing_error_orders=("reason_lc", lambda x: x.str.contains("listing error").sum()),
                oj_requested_orders=("reason_lc", lambda x: x.str.contains("oj requested").sum()),
                high_shipping_cost_orders=("reason_lc", lambda x: x.str.contains("high shipping cost").sum()),

                # -------------------
                # Items / products
                # -------------------
                total_items=("itemid", "nunique"),
                total_products=("pid", "nunique"),

                # -------------------
                # Availability / stock
                # -------------------
                #disabled_items=("availability", lambda x: (x > 2).sum()),
                #ltl_items = ("isLTL", lambda x: (x == 1).sum()),
                #fragile_items = ("fragility", lambda x: (x == "Fragile").sum()), 
                disabled_items=("itemid",
                    lambda x: x[df.loc[x.index, "availability"] > 2].nunique()
                ),

                ltl_items=("itemid",
                    lambda x: x[df.loc[x.index, "isLTL"] == 1].nunique()
                ),

                fragile_items=("itemid",
                    lambda x: x[df.loc[x.index, "fragility"] == "Fragile"].nunique()
                ),


                shipped_orders = ("shipped_ind", lambda x: (x == 1).sum()),
                delivered_orders = ("delivered_ind", lambda x: (x == 1).sum()),
                actual_ship_avg_days = ("ship_order_days_actual","mean"),
                actual_delivery_avg_days = ("delivery_order_days_actual","mean"),
                cancellation_promised_ship_days_avg = ("cancellation_ship_days","mean"),
                cancellation_promised_ship_days_median = ("cancellation_ship_days","median"),
                cancellation_promised_delivery_days_avg = ("cancellation_delivery_days","mean"),
                cancellation_promised_delivery_days_median = ("cancellation_delivery_days","median"),
                ship_late_orders = (
                    "ship_status",lambda x: x.isin(["Late", "Late – Cancelled", "Cancelled After Promised"]).sum()),
                ship_missing_orders = ("ship_status", lambda x: (x == "Actual Missing Past Promised").sum()),
                delivery_late_orders = (
                    "delivery_status",lambda x: x.isin(["Late", "Late – Cancelled", "Cancelled After Promised"]).sum()),
                delivery_missing_orders = ("delivery_status", lambda x: (x == "Actual Missing Past Promised").sum()),
                ship_early_orders = (
                    "ship_status",lambda x: x.isin(["Early", "On-time"]).sum()),
                delivery_early_orders = (
                    "delivery_status",lambda x: x.isin(["Early", "On-time"]).sum()),    
                unavailable_at_order=("status_at_time_of_order_cat_updated",
                                    lambda x: (x == "Unavailable").sum()),
                available_at_order=("status_at_time_of_order_cat_updated",
                                    lambda x: (x == "Available").sum()),
                qty_insufficient_fulfill_order = ("can_fulfill_order",
                                    lambda x: (x == "Insufficient").sum()),                       
                avg_stock_at_order=("status_at_time_of_order_qty", "mean"),
                #leadtime_delayed_or_missing_orders = ("leadtime_status", lambda x: (x == "Delayed|Missing").sum())
                leadtime_delayed_or_missing_orders = (
                    "leadtime_status",lambda x: x.isin(["Delayed", "Missing"]).sum()),
                leadtime_on_time_orders = (
                    "leadtime_status",lambda x: x.isin(["On Time"]).sum()),
                vendor_disabled=("vendor_disabled", "last"),
                naomi_vendor=("naomi_vendor", "last")
            )
            .reset_index()
        )

        vendor_summary["avg_stock_at_order"] = (
            vendor_summary["avg_stock_at_order"]).round(3)

        vendor_summary["seller_cancel_rate"] = (
            vendor_summary["seller_cancel_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["customer_cancel_rate"] = (
            vendor_summary["customer_cancel_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["safe_order_rate"] = (
            vendor_summary["safe_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["unavailable_ratio"] = (
            vendor_summary["unavailable_at_order"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["available_ratio"] = (
            vendor_summary["available_at_order"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["leadtime_prblm_ratio"] = (
            vendor_summary["leadtime_delayed_or_missing_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["leadtime_on_time_ratio"] = (
            vendor_summary["leadtime_on_time_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["disabled_items_ratio"] = (
            vendor_summary["disabled_items"] / vendor_summary["total_items"]
        ).round(3)

        vendor_summary["ltl_items_ratio"] = (
            vendor_summary["ltl_items"] / vendor_summary["total_items"]
        ).round(3)

        vendor_summary["fragile_items_ratio"] = (
            vendor_summary["fragile_items"] / vendor_summary["total_items"]
        ).round(3)

        vendor_summary["expected_late_ratio"] = (
            vendor_summary["expected_late_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["backorder_ratio"] = (
            vendor_summary["backorder_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["discontinued_ratio"] = (
            vendor_summary["discontinued_orders"] / vendor_summary["total_orders"]
        ).round(3)


        vendor_summary["pricing_error_ratio"] = (
            vendor_summary["pricing_issue_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["listing_error_ratio"] = (
            vendor_summary["listing_error_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["oj_req_ratio"] = (
            vendor_summary["oj_requested_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["high_ship_cost_ratio"] = (
            vendor_summary["high_shipping_cost_orders"] / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["ship_issue_ratio"] = (
            (vendor_summary["ship_late_orders"] + vendor_summary["ship_missing_orders"]) / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["delivery_issue_ratio"] = (
            (vendor_summary["delivery_late_orders"] + vendor_summary["delivery_missing_orders"]) / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["ship_safe_ratio"] = (
            (vendor_summary["ship_early_orders"]) / vendor_summary["total_orders"]
        ).round(3)

        vendor_summary["delivery_safe_ratio"] = (
            (vendor_summary["delivery_early_orders"]) / vendor_summary["total_orders"]
        ).round(3)        

        vendor_summary["total_orders_weight"] = (vendor_summary["total_orders"]/vendor_summary["total_orders"].sum()).round(4)

        ## code deleted from here of vendor risk features



        vendor_summary_subset = vendor_summary[["vendor_name","naomi_vendor","vendor_disabled",#"avg_stock_at_order"
        #"unavailable_ratio",#"available_ratio", "leadtime_prblm_ratio",
        #"leadtime_on_time_ratio",#"seller_cancel_rate",#"customer_cancel_rate"
        "safe_order_rate",
        #"disabled_items_ratio",
                                                #"ltl_items_ratio",
                                                "total_orders_weight",#"fragile_items_ratio",
                                                #"expected_late_ratio","backorder_ratio",#"discontinued_ratio","pricing_error_ratio","listing_error_ratio","oj_req_ratio",
                                                #"high_ship_cost_ratio",
                                                #"ship_issue_ratio","delivery_issue_ratio",
                                                "ship_safe_ratio","delivery_safe_ratio"
                                                #,"actual_ship_avg_days","actual_delivery_avg_days","cancellation_promised_ship_days_avg","cancellation_promised_ship_days_median"
                                                #,"cancellation_promised_delivery_days_avg","cancellation_promised_delivery_days_median"
                                                ]]

        vendor_monthly_risk_subset_1 = vendor_monthly_risk[["vendor_name","overall_SCP",
        "overall_SCC",
        "overall_risk_indicator"#,"overall_OR_risk_flag"#,"overall_RR_risk_flag"
        ]]

        #vendor_risk_features_subset = vendor_risk_features[["vendor_name",#"SCP_median","SCP_std","SCP_CV",#"SCP_p90",
        #"SCP_trend","SCP_ewm",
                                                            #"SCC_median","SCC_std","SCC_CV",#"SCC_p90",
                                                            #"SCC_trend","SCC_ewm",
                                                            #"risk_indicator_risk_prop",
                                                            #"risk_indicator_risk_weighted_prop",
                                                            #"OR_risk_prop",
                                                            #"OR_risk_weighted_prop",
                                                            #"RR_risk_prop",
                                                            #"RR_risk_weighted_prop"
                                                            #]]

        vendor_wise_final = vendor_summary_subset.merge(vendor_monthly_risk_subset_1, on = "vendor_name",how = "left")
        #vendor_wise_final = vendor_wise_final.merge(vendor_risk_features_subset, on = "vendor_name", how = "left")

        vendor_wise_final["overall_SCP"] = (vendor_wise_final["overall_SCP"]/100).round(4)
        #vendor_wise_final["overall_SCC"] = (vendor_wise_final["overall_SCC"]/100).round(4)

        vendor_wise_final["overall_SCC"] = (vendor_wise_final["overall_SCC"]/100).round(4)

        from sklearn.preprocessing import StandardScaler
        df = vendor_wise_final.copy()

        features = df.drop(columns=[
            'vendor_name','naomi_vendor','vendor_disabled',"total_orders_weight"
            #'overall_risk_indicator','overall_OR_risk_flag','overall_RR_risk_flag'
        ])

        features = features.loc[:, features.std() > 1e-6]

        corr = features.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        #features = features.drop(columns=[
        #    col for col in upper.columns if any(upper[col] > 0.95)
        #    ])

        features = features.fillna(0.01)

        #features_aligned = features.copy()

        # Invert GOOD metrics so higher = worse
        #features["avg_stock_at_order"] *= -1
        #features["cancellation_promised_ship_days_avg"] *= -1

        X_scaled = StandardScaler().fit_transform(features)

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import calinski_harabasz_score
        from kneed import KneeLocator
        from factor_analyzer.factor_analyzer import (
            calculate_kmo,
            calculate_bartlett_sphericity
        )
        from semopy import Model


        def select_number_of_factors(
            X_scaled,
            features_df,
            max_factors=8,
            pa_iter=100,
            variance_thresholds=(0.7, 0.8),
            plot=True,
            random_state=42
        ):
            """
            NbClust-like factor selection with:
            - Scree + elbow
            - Parallel analysis
            - Kaiser
            - Variance explained
            - Pseudo-F (Calinski–Harabasz on factor scores)
            - Voting-based recommendation
            """

            np.random.seed(random_state)

            # ---------------------------
            # 1️⃣ PCA eigenvalues
            # ---------------------------
            pca_full = PCA()
            pca_full.fit(X_scaled)

            eigenvalues = pca_full.explained_variance_
            explained_ratio = pca_full.explained_variance_ratio_
            cum_var = np.cumsum(explained_ratio)

            # ---------------------------
            # 2️⃣ Scree elbow
            # ---------------------------
            elbow = KneeLocator(
                range(1, len(eigenvalues) + 1),
                eigenvalues,
                curve="convex",
                direction="decreasing"
            ).elbow

            # ---------------------------
            # 3️⃣ Parallel Analysis
            # ---------------------------
            n_obs, n_vars = X_scaled.shape
            rand_eigs = np.zeros((pa_iter, n_vars))

            for i in range(pa_iter):
                X_rand = np.random.normal(size=(n_obs, n_vars))
                pca_rand = PCA()
                pca_rand.fit(X_rand)
                rand_eigs[i, :] = pca_rand.explained_variance_

            rand_eig_95 = np.percentile(rand_eigs, 95, axis=0)
            n_factors_pa = int(np.sum(eigenvalues > rand_eig_95))

            # ---------------------------
            # 4️⃣ Kaiser rule
            # ---------------------------
            n_factors_kaiser = int(np.sum(eigenvalues > 1))

            # ---------------------------
            # 5️⃣ Variance explained
            # ---------------------------
            var_based = {
                f"{int(v*100)}%": int(np.argmax(cum_var >= v) + 1)
                for v in variance_thresholds
            }

            # ---------------------------
            # 6️⃣ Pseudo-F statistic (Calinski–Harabasz)
            # ---------------------------
            pseudo_f_scores = {}

            for k in range(2, min(max_factors + 1, X_scaled.shape[1] + 1)):
                pca_k = PCA(n_components=k, random_state=random_state)
                scores_k = pca_k.fit_transform(X_scaled)

                labels = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=random_state
                ).fit_predict(scores_k)

                pseudo_f_scores[k] = calinski_harabasz_score(scores_k, labels)

            best_pseudo_f = max(pseudo_f_scores, key=pseudo_f_scores.get)

            # ---------------------------
            # 7️⃣ KMO & Bartlett
            # ---------------------------
            _, kmo_model = calculate_kmo(features_df)
            _, bartlett_p = calculate_bartlett_sphericity(features_df)

            # ---------------------------
            # 8️⃣ Voting (NbClust-style)
            # ---------------------------
            votes = []

            if elbow is not None:
                votes.append(elbow)

            votes.extend([
                n_factors_pa,
                n_factors_kaiser,
                best_pseudo_f
            ])

            votes.extend(var_based.values())

            #vote_series = pd.Series(votes)
            #recommended = int(vote_series.mode()[0])

            vote_series = pd.Series(votes)
            vote_counts = vote_series.value_counts()

            # If all votes have the same frequency → take median
            if vote_counts.nunique() == 1:
                recommended = int(np.median(vote_series))
            else:
                recommended = int(vote_counts.idxmax())    

            # ---------------------------
            # 9️⃣ Plots
            # ---------------------------
            fig = None
            if plot:
                fig, ax = plt.subplots(1, 3, figsize=(18, 5))

                # Scree plot
                ax[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o")
                ax[0].axhline(1, color="red", linestyle="--", label="Eigenvalue = 1")
                if elbow:
                    ax[0].axvline(elbow, color="green", linestyle="--", label=f"Elbow = {elbow}")
                ax[0].set_title("Scree Plot")
                ax[0].legend()

                # Parallel analysis
                ax[1].plot(eigenvalues, marker="o", label="Observed")
                ax[1].plot(rand_eig_95, marker="x", linestyle="--", label="Random 95%")
                ax[1].set_title("Parallel Analysis")
                ax[1].legend()

                # Pseudo-F
                ax[2].plot(
                    list(pseudo_f_scores.keys()),
                    list(pseudo_f_scores.values()),
                    marker="o"
                )
                ax[2].axvline(
                    best_pseudo_f,
                    color="green",
                    linestyle="--",
                    label=f"Best k = {best_pseudo_f}"
                )
                ax[2].set_title("Pseudo-F (Calinski–Harabasz)")
                ax[2].set_xlabel("Number of Factors")
                ax[2].legend()

                plt.tight_layout()
                #plt.show()

            # ---------------------------
            # 🔟 Summary
            # ---------------------------
            summary = {
                "recommended_factors": recommended,
                "elbow_factors": elbow,
                "parallel_analysis_factors": n_factors_pa,
                "kaiser_factors": n_factors_kaiser,
                "variance_based_factors": var_based,
                "pseudo_f_best": best_pseudo_f,
                "pseudo_f_scores": pseudo_f_scores,
                "kmo": round(kmo_model, 3),
                "bartlett_p_value": bartlett_p,
                "votes": vote_series.value_counts().to_dict()
            }

            return summary, fig

        feature_selection_summary, feature_selection_fig = select_number_of_factors(
            X_scaled=X_scaled,
            features_df=features,
            max_factors=8,
            plot=True
        )

        #summary

        recommended_factors = feature_selection_summary["recommended_factors"]

        from factor_analyzer import FactorAnalyzer

        max_factors = recommended_factors

        fa = FactorAnalyzer(n_factors=max_factors, rotation='varimax')#,method = 'principal')
        fa.fit(X_scaled)
        

        loadings = pd.DataFrame(
            fa.loadings_,
            index=features.columns,
            columns=[f'Factor{i+1}' for i in range(max_factors)]
        )

        factor_scores = fa.transform(X_scaled)

        #for i, col in enumerate(loadings.columns):
        #    if loadings[col].sum() < 0:
        #        loadings[col] *= -1
        #        factor_scores[:, i] *= -1



        from sklearn.decomposition import PCA

        #pc_scores = []

        #pca = PCA(n_components=2)
        #pc_scores = pca.fit_transform(factor_scores)

        #explained_var = []

        #explained_var = pca.explained_variance_ratio_

        loadings_df = loadings.copy()

        def assign_factor_absmax(row):
            return row.abs().idxmax()

        loadings_df["assigned_factor"] = loadings_df.apply(assign_factor_absmax, axis=1)


        # Communalities
        communalities = (loadings ** 2).sum(axis=1)

        communalities_df = communalities.to_frame(name="communality")
        #communalities_df

        # Eigenvalues
        eigenvalues = (loadings ** 2).sum(axis=0)

        eigenvalues_df = eigenvalues.to_frame(name="eigenvalue")
        #eigenvalues_df

        # Percentage variance explained
        total_variance = eigenvalues.sum()

        var_explained_pct = (eigenvalues / total_variance) * 100

        var_explained_df = pd.DataFrame({
            "eigenvalue": eigenvalues,
            "variance_explained_pct": var_explained_pct,
            "cumulative_variance_pct": var_explained_pct.cumsum()
        })

        #var_explained_df

        fa_summary = (
            loadings_df
            .join(communalities_df)
            .reset_index()
            .rename(columns={"index": "variable"})
        )

        #fa_summary

        factor_summary = var_explained_df.reset_index().rename(
            columns={"index": "factor"}
        )

        #factor_summary




        #import numpy as np

        #loadings_df = loadings.copy()  # your factor loadings DataFrame

        #threshold = 0.3
        #cross_threshold = 0.3

        #threshold = 0.3
        #diff_threshold = 0.15

        #def assign_factor(row):
        #    abs_row = row.abs()
        #    strong = abs_row[abs_row >= threshold]

            # No strong loading
        #    if len(strong) == 0:
        #        return "Unassigned"

            # More than one strong loading → check difference
        #    if len(strong) > 1:
        #        sorted_vals = strong.sort_values(ascending=False)
        #        if (sorted_vals.iloc[0] - sorted_vals.iloc[1]) < diff_threshold:
        #            return "Cross-loading"
        #        else:
        #            return sorted_vals.index[0]

            # Exactly one strong loading
        #    return strong.idxmax()



        #loadings_df["assigned_factor"] = loadings_df.apply(assign_factor, axis=1)

        #from sklearn.preprocessing import StandardScaler
        #from factor_analyzer import FactorAnalyzer

        #features_current = features.copy()
        #max_iter = 10

        #for iteration in range(1, max_iter + 1):

        #    print(f"\n🔁 Iteration {iteration}")
            
            # Scale
        #    X_scaled = StandardScaler().fit_transform(features_current)

            # Fit FA
        #    fa = FactorAnalyzer(
        #        n_factors=recommended_factors,
        #        rotation="varimax",
        #        method="principal"
        #    )
        #    fa.fit(X_scaled)

        #    loadings = pd.DataFrame(
        #        fa.loadings_,
        #        index=features_current.columns,
        #        columns=[f"Factor{i+1}" for i in range(recommended_factors)]
        #    )

            # Assign factors
        #    loadings_df = loadings.copy()
        #    loadings_df["assigned_factor"] = loadings_df.apply(assign_factor, axis=1)

            # Check problematic variables
        #    bad_vars = loadings_df.loc[
        #        loadings_df["assigned_factor"].isin(["Unassigned", "Cross-loading"])
        #    ].index.tolist()

        #    print(f"❌ Unassigned/Cross-loading vars: {len(bad_vars)}")

            # ✅ Converged
        #    if len(bad_vars) == 0:
        #        print("✅ All variables cleanly assigned.")
        #        break

            # 🛑 Stop if nothing changes
        #    if set(bad_vars) == set(features_current.columns):
        #        print("🛑 No improvement possible. Stopping.")
        #        break

            # Drop bad variables
        #    features_current = features_current.drop(columns=bad_vars)

            # Safety check
        #    if features_current.shape[1] < recommended_factors + 1:
        #        print("🛑 Too few variables left to sustain factors. Stopping.")
        #        break
        
        #features_reduced = features_current

        #assigned_vars = loadings_df.loc[
        #    ~loadings_df["assigned_factor"].isin(["Unassigned", "Cross-loading"])
        #].index.tolist()

        #features_reduced = features[assigned_vars]

        #from sklearn.preprocessing import StandardScaler
        #from factor_analyzer import FactorAnalyzer

        #X_scaled_reduced = StandardScaler().fit_transform(features_reduced)

        #fa = FactorAnalyzer(
        #    n_factors=recommended_factors,
        #    rotation="varimax",
        #    method="principal"   # more stable for SEM prep
        #)
        #fa.fit(X_scaled_reduced)

        #loadings = pd.DataFrame(
        #    fa.loadings_,
        #    index=features_reduced.columns,
        #    columns=[f"Factor{i+1}" for i in range(recommended_factors)]
        #)

        #factor_scores = fa.transform(X_scaled_reduced)

        #loadings_df = loadings.copy()
        #loadings_df["assigned_factor"] = loadings_df.apply(assign_factor, axis=1)

        var_to_factor = (
            loadings_df["assigned_factor"]
            .to_dict()
        )

        import ast
        import time
        import google.generativeai as genai
        from google.api_core.exceptions import (
            ResourceExhausted,
            PermissionDenied,
            InternalServerError,
            ServiceUnavailable
        )

        # ---------------------------------
        # PROMPT (locked format)
        # ---------------------------------
        prompt = f"""
        You are a data science assistant helping to name latent risk factors.

        Input:
        You will be given a Python dictionary mapping observed variables to factor IDs.
        Each factor represents a latent risk driver inferred from factor analysis.

        Task:
        1. For each factor (e.g., Factor1, Factor2, …), analyze the variables assigned to it.
        2. Infer the common business / operational theme represented by those variables.
        3. Generate a concise, meaningful, professional factor name that:
        - Reflects the shared risk concept
        - Uses only letters, numbers, and underscores (SEM-safe)
        - Ends with _Risk, _Risk_Indicator, or _Weight where appropriate
        - Is suitable for vendor risk modeling

        Output format:
        Return ONLY a valid Python dictionary named factor_names.

        Input dictionary:
        {var_to_factor}
        """

        # ---------------------------------
        # CONFIG
        # ---------------------------------
        API_KEYS = [
                "AIzaSyDeYXu0cd1wVWoD-j7oMPoQ-P-a7MCPa-w",
            "AIzaSyD0c8_xh7SSxJ4EslADranceSw7Di-ZPJw",
            "AIzaSyAWWBiE7hP9QEgALq8UzFFYLkwdjv5RjUc",
            "AIzaSyDGWjXkz1gwY_ZfTJGtJZ3UgzTcgk5Gfw8",
            "AIzaSyDZikyW5ewncExiPnykA5Ejc6tOi2UL-9o",
            "AIzaSyBb5JXU4VnvHvZXNt85dznODVC6MgmjSZg",
            "AIzaSyCXH74VLKgVl5r44gusFGy5JVbOzdTC1x0",
            "AIzaSyDE2kDIoi9GjrK0tEdOAlmXgNg9rmTaigU",
            "AIzaSyBYKJVU0SExJtAl4_uIMEgPQiiecEHNAuc",
            "AIzaSyCY5Ct4OeJA8fKxGJiZ-8N2-1-VcaWNY74"
        ]

        MODEL_NAME = "gemini-2.5-flash"
        MAX_RETRIES_PER_KEY = 3
        RETRY_SLEEP_SECONDS = 2


        # ---------------------------------
        # GEMINI CALL WITH FAILOVER
        # ---------------------------------
        def get_factor_names_with_fallback(prompt, api_keys):
            last_error = None

            for key_idx, api_key in enumerate(api_keys, start=1):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(MODEL_NAME)

                for attempt in range(1, MAX_RETRIES_PER_KEY + 1):
                    try:
                        print(f"🔑 Key {key_idx} | Attempt {attempt}")

                        response = model.generate_content(prompt)
                        text = response.text.strip()

                        import re
                        import ast

                        def extract_factor_names(text):
                            # Remove markdown code fences
                            text = text.replace("```python", "").replace("```", "").strip()

                            # Extract dictionary content
                            match = re.search(r"\{.*\}", text, re.DOTALL)
                            if not match:
                                raise ValueError(f"No dictionary found in Gemini output:\n{text}")

                            dict_str = match.group(0)

                            return ast.literal_eval(dict_str)
                        


                        print("---- RAW GEMINI OUTPUT ----")
                        print(text)
                        print("---------------------------")

                        factor_names = extract_factor_names(text)
                                        
                        print("✅ Gemini call successful")
                        return factor_names

                    # ----- quota / auth → switch key -----
                    except (ResourceExhausted, PermissionDenied) as e:
                        print(f"⚠️ Key {key_idx} exhausted or unauthorized. Switching key.")
                        last_error = e
                        break  # move to next key

                    # ----- transient errors → retry same key -----
                    except (InternalServerError, ServiceUnavailable, TimeoutError) as e:
                        print(f"⏳ Transient error on key {key_idx}, retrying...")
                        last_error = e
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue

                    # ----- unexpected errors -----
                    except Exception as e:
                        print(f"❌ Unexpected error on key {key_idx}: {e}")
                        last_error = e
                        break

            raise RuntimeError("All Gemini API keys failed.") from last_error


        # ---------------------------------
        # RUN
        # ---------------------------------
        factor_names = get_factor_names_with_fallback(prompt, API_KEYS)

        print("\nGenerated factor names:")
        print(factor_names)

        outcome_var = "overall_SCP"

        from collections import defaultdict

        factor_to_vars = defaultdict(list)

        for var, fac in var_to_factor.items():
            factor_to_vars[fac].append(var)

        latent_factors = {
            fac: vars_
            for fac, vars_ in factor_to_vars.items()
            if len(vars_) >= 2
        }

        covariate_factors = {
            fac: vars_
            for fac, vars_ in factor_to_vars.items()
            if len(vars_) == 1
        }
        
        measurement_lines = []

        for fac, vars_ in latent_factors.items():
            latent_name = factor_names[fac]
            rhs = " + ".join(vars_)
            measurement_lines.append(f"{latent_name} =~ {rhs}")


        structural_terms = []

        # latent predictors
        for fac in latent_factors:
            structural_terms.append(factor_names[fac])

        # covariates (single-indicator factors)
        for fac, vars_ in covariate_factors.items():
            structural_terms.extend(vars_)

        structural_line = f"{outcome_var} ~ " + " + ".join(structural_terms)

        model_desc = (
            "\n\n".join(measurement_lines)
            + "\n\n"
            + structural_line
        )

        print(model_desc)

        #from semopy import Model

        model = Model(model_desc)
        res = model.fit(df)
        model_summary = model.inspect(std_est=True)

        from semopy import Model
        from semopy import stats



        fit_stats = stats.calc_stats(model)
        #fit_stats = model.calc_stats()
        #fit_stats

        sem_metrics = {
            "CFI": round(fit_stats["CFI"], 3),
            "TLI": round(fit_stats["TLI"], 3),
            "RMSEA": round(fit_stats["RMSEA"], 3),
            #"SRMR": round(fit_stats["SRMR"], 3),
            "Chi2": round(fit_stats["chi2"], 2),
            "Chi2_pval": round(fit_stats["chi2 p-value"], 2),
            "df": fit_stats["DoF"],
            "Chi2_df": round(fit_stats["chi2"] / fit_stats["DoF"], 2),
            "AIC": round(fit_stats["AIC"], 2),
            "BIC": round(fit_stats["BIC"], 2)
        }

        #pd.DataFrame.from_dict(sem_metrics, orient="index", columns=["Value"])

        est = model.inspect(std_est=True)

        sem_paths = est[
            (est["op"] == "~") &
            (est["lval"] == outcome_var)
        ][["rval", "Estimate"]]

        latent_predictors = [
            factor_names[fac] for fac in latent_factors
        ]

        covariate_predictors = [
            factor_to_vars[fac][0] for fac in covariate_factors
        ]

        latent_weights = (
            sem_paths[sem_paths["rval"].isin(latent_predictors)]
            .set_index("rval")["Estimate"]
        )


        #latent_weights = sem_paths[
        #    sem_paths["rval"].isin([
        #        "Fulfilment_Delay_Cancellation_Timing_Risk",
        #        "Delivery_Execution_Customer_Impact_Risk",
        #        "Supply_Operational_Risk_Indicator"
        #    ])
        #].set_index("rval")["Estimate"]

        observed_weights = (
            sem_paths[sem_paths["rval"].isin(covariate_predictors)]
            .set_index("rval")["Estimate"]
        )

        #observed_weights = sem_paths[
        #    sem_paths["rval"].isin([
        #        "total_orders_weight",
        #        "unavailable_ratio"
        #    ])
        #].set_index("rval")["Estimate"]

        latent_weights = latent_weights[~latent_weights.index.duplicated(keep="first")]
        observed_weights = observed_weights[~observed_weights.index.duplicated(keep="first")]

        positive_vars = {
            "safe_order_rate",
            "ship_safe_ratio",
            "delivery_safe_ratio"
        }

        negative_vars = {
            "overall_SCP",
            "overall_SCC",
            "overall_risk_indicator"
        }

        import numpy as np

        factor_signs = {}

        for fac in loadings.columns:
            # loadings for this factor
            fac_loads = loadings[fac]

            # contribution from negative indicators
            neg_contrib = fac_loads.loc[
                fac_loads.index.intersection(negative_vars)
            ].mean()

            # contribution from positive indicators
            pos_contrib = fac_loads.loc[
                fac_loads.index.intersection(positive_vars)
            ].mean()

            #"""
            #We want:
            #- negative factor score = higher risk
            #So flip if:
            #- negative indicators load positively
            #- OR positive indicators load negatively
            #"""
            if neg_contrib > 0 or pos_contrib < 0:
                factor_signs[fac] = -1
            else:
                factor_signs[fac] = 1
        
        factor_scores_signed = factor_scores.copy()

        for i, fac in enumerate(loadings.columns):
            factor_scores_signed[:, i] *= factor_signs[fac]
        

        # map factor indices to SEM-safe names
        factor_order = list(latent_weights.index)

        factor_score_df = pd.DataFrame(
            factor_scores_signed,
            columns=loadings.columns,
            index=df.index
        )

        # rename Factor1 → Fulfilment_Delay_Risk, etc.
        factor_score_df = factor_score_df.rename(columns=factor_names)

        # attach to df
        df = pd.concat([df, factor_score_df], axis=1)

        df["vendor_risk_score"] = factor_scores_signed.sum(axis=1)

        df["vendor_risk_score_final"] = (df["vendor_risk_score"]*(df["total_orders_weight"]*100)).round(4)

        mean_score = df["vendor_risk_score_final"].mean()
        std_score = df["vendor_risk_score_final"].std(ddof=0)

        df["vendor_risk_z"] = (
            (df["vendor_risk_score_final"] - mean_score) / std_score
        ).round(4)

        min_val = df["vendor_risk_score_final"].min()
        max_val = df["vendor_risk_score_final"].max()

        df["vendor_risk_minmax"] = (
            (df["vendor_risk_score_final"] - min_val) /
            (max_val - min_val)
        ).round(4)

        # Optional: convert to 0–100 scale
        df["vendor_risk_index_0_100"] = (df["vendor_risk_minmax"] * 100).round(2)

        df["vendor_risk_index_0_100"] = (100 - df["vendor_risk_index_0_100"])

        factor_scores_df = pd.DataFrame(
            factor_scores,
            columns=loadings.columns,   # Factor1, Factor2, ...
            index=df.index
        )        

        factor_scores_1 = pd.concat(
            [df[["vendor_name"]], factor_scores_df],
            axis=1
        )

        factor_scores_signed_df = pd.DataFrame(
            factor_scores_signed,
            columns=loadings.columns,   # Factor1, Factor2, ...
            index=df.index
        )        

        factor_scores_signed_1 = pd.concat(
            [df[["vendor_name"]], factor_scores_signed_df],
            axis=1
        )


        df_to_display = df[["vendor_name","naomi_vendor","vendor_disabled","vendor_risk_score_final",
        "vendor_risk_z","vendor_risk_index_0_100","safe_order_rate","ship_safe_ratio","delivery_safe_ratio",
        "overall_SCP","overall_SCC","overall_risk_indicator","total_orders_weight"]]

        from scipy.stats import pearsonr

        target = "vendor_risk_index_0_100"

        corr_vars = [
            "safe_order_rate",
            "ship_safe_ratio",
            "delivery_safe_ratio",
            "overall_SCP",
            "overall_SCC"
        ]

        corr_results = {}

        for var in corr_vars:
            valid = df_to_display[[target, var]].dropna()

            if len(valid) > 2:
                r, p = pearsonr(valid[target], valid[var])
                corr_results[var] = {
                    "correlation": round(r, 4),
                    "p_value": round(p, 6),
                    "n_obs": len(valid)
                }

        group0 = df_to_display[df_to_display["overall_risk_indicator"] == 0][target].dropna()
        group1 = df_to_display[df_to_display["overall_risk_indicator"] == 1][target].dropna()

        from scipy.stats import shapiro

        normal_0 = shapiro(group0)[1] > 0.05 if len(group0) >= 3 else False
        normal_1 = shapiro(group1)[1] > 0.05 if len(group1) >= 3 else False

        is_normal = normal_0 and normal_1

        from scipy.stats import ttest_ind, mannwhitneyu

        if is_normal:
            stat, p_val = ttest_ind(group0, group1, equal_var=False)
            test_used = "Welch t-test"
        else:
            stat, p_val = mannwhitneyu(group0, group1, alternative="two-sided")
            test_used = "Mann–Whitney U"

        binary_test_result = {
            "variable": "overall_risk_indicator",
            "test_used": test_used,
            "mean_risk_indicator_0": round(group0.mean(), 4),
            "mean_risk_indicator_1": round(group1.mean(), 4),
            "p_value": round(p_val, 6),
            "n_0": len(group0),
            "n_1": len(group1)
        }

        analysis_results = {
            "correlations": corr_results,
            "binary_test_overall_risk_indicator": binary_test_result
        }        

        df_to_display = df_to_display.sort_values(
            by="vendor_risk_score_final",
            ascending=True
            )

        # --------------------------------------------------------------------------
        # 6.a Dashboard: Analysis Diagnostics
        # --------------------------------------------------------------------------
        with st.expander("Analysis Details & Diagnostics", expanded=False):
            st.markdown("### 1. Optimal Number of Factors Selection")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**Selection Summary**")
                st.json(feature_selection_summary)
            with col2:
                if feature_selection_fig:
                    st.write("**Eigenvalue Plots**")
                    st.pyplot(feature_selection_fig)
            
            st.markdown("---")
            st.markdown("### 2. Factor Analysis Metrics and loadings")
            col3, col4,col5 = st.columns(3)
            with col3:
                st.write("**Communalities**")
                st.dataframe(communalities_df, use_container_width=True)
            with col4:
                st.write("**Variance Explained**")
                st.dataframe(factor_summary, use_container_width=True)
            with col5:
                st.write("**Factor loadings**")
                st.write(loadings_df)
                        
            st.markdown("---")
            st.markdown("### 3. Factor Interpretation")
            col6, col7,col8 = st.columns(3)
            with col6:
                st.write("**Variable to Factor Mapping**")
                st.write(var_to_factor)
            with col7:
                st.write("**Generated Factor Names**")
                st.write(factor_names)
            with col8:
                st.write("**Factor Signs**")
                st.write(factor_signs)
                
            st.markdown("---")
            st.markdown("### 4. Factor Scores Comparison")
            col9, col10 = st.columns(2)
            with col9:
                st.write("**Old Factor Scores**")
                st.write(factor_scores_1)
            with col10:
                st.write("**New Sign Factor Scores**")
                st.write(factor_scores_signed_1)

            st.markdown("---")
            st.markdown("### 5. Factor Score Validation")

            col11, col12 = st.columns(2)

            with col11:
                st.write("**Correlation with Vendor Risk Index (Continuous Drivers)**")

                corr_df = (
                    pd.DataFrame(analysis_results["correlations"])
                    .T
                    .reset_index()
                    .rename(columns={"index": "variable"})
                )

                st.dataframe(corr_df, use_container_width=True)

            with col12:
                st.write("**Risk Index Comparison by Overall Risk Indicator**")

                binary_df = pd.DataFrame(
                    [analysis_results["binary_test_overall_risk_indicator"]]
                )

                st.dataframe(binary_df, use_container_width=True)


            st.markdown("---")
            st.markdown("### 6. SEM Model Results")
            st.write("**Model Description**")
            st.code(model_desc, language="text")


            col13, col14 = st.columns(2)
            with col13:
                st.write("**Fit Metrics**")
                st.write(sem_metrics)
            with col14:
                st.write("**Model Parameter Estimates**")
                st.dataframe(est, use_container_width=True)

        #anchor_var = "overall_SCP"
        #val = np.corrcoef(df[fac], df[anchor_var])[0,1]

        #for fac in latent_weights.index:
        #    if np.corrcoef(df[fac], df[anchor_var])[0,1] < 0:
        #        df[fac] *= -1

        #df["vendor_risk_score"] = 0

        # latent factor contribution
        #for fac, w in latent_weights.items():
        #    df["vendor_risk_score"] += df[fac] * w

        # observed covariate contribution
        #for var, w in observed_weights.items():
        #    df["vendor_risk_score"] += df[var] * w            

        
        #df["vendor_risk_score_norm"] = (
        #    (df["vendor_risk_score"] - df["vendor_risk_score"].min()) /
        #    (df["vendor_risk_score"].max() - df["vendor_risk_score"].min())
        #) * 100       

        #df["total_orders_weight_norm"] = (
        #    (df["total_orders_weight"] - df["total_orders_weight"].min()) /
        #    (df["total_orders_weight"].max() - df["total_orders_weight"].min())
        #)

        #df["vendor_risk_score_weighted"] = (
        #    df["vendor_risk_score_norm"] * df["total_orders_weight_norm"]
        #)

        #df["vendor_risk_score_weighted_norm"] = (
        #    (df["vendor_risk_score_weighted"] - df["vendor_risk_score_weighted"].min()) /
        #    (df["vendor_risk_score_weighted"].max() - df["vendor_risk_score_weighted"].min())
        #) * 100

        #contrib = pd.DataFrame(index=df.index)

        #for fac, w in latent_weights.items():
        #    contrib[fac] = df[fac] * w

        #for var, w in observed_weights.items():
        #    contrib[var] = df[var] * w

        #contrib_pct = contrib.div(contrib.sum(axis=1), axis=0) * 100

        #contrib_pct = contrib_pct.add_suffix("_contrib_pct")

        #df = pd.concat([df, contrib_pct], axis=1)



        #df = df.reset_index(drop=True)

    # --------------------------------------------------------------------------
    # 6. Final Table Display & Filters
    # --------------------------------------------------------------------------
    st.subheader("Vendor Analysis Results")
    st.write("Use the filters below to refine the vendor list.")

    col1, col2 = st.columns(2)
    with col1:
        # Toggles
        filter_naomi = st.checkbox("Show Only Non-Naomi Vendors (naomi_vendor = False)", value=False)
    with col2:
        filter_disabled = st.checkbox("Show Only Non-Disabled Vendors (vendor_disabled = False)", value=False)

    #display_df = vendor_wise_final.copy()
    display_df = df_to_display.copy()

    # Apply Filters
    if filter_naomi:
        display_df = display_df[display_df["naomi_vendor"] != True]
    
    if filter_disabled:
        display_df = display_df[display_df["vendor_disabled"] != True]

    st.write(f"Showing {len(display_df)} vendors.")
    st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()



