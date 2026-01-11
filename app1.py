# =============================================================
# ENTERPRISE AI MARKET INTELLIGENCE
# TAB 1 & TAB 2 (FINAL, ERROR-FREE)
# =============================================================

# ================== CORE IMPORTS ==================
import streamlit as st
import pandas as pd
import numpy as np
import pickle  # <-- THIS FIXES YOUR ERROR
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest

# ================== FORECASTING ==================
from prophet import Prophet
import yfinance as yf

# ================== SENTIMENT ==================
from wordcloud import WordCloud

# ================== REPORTING ==================
from fpdf import FPDF
import tempfile
import os


# ------------------------------------------------------------
# LOAD SENTIMENT MODELS (GLOBAL, CACHED)
# ------------------------------------------------------------
@st.cache_resource
def load_sentiment_models():
    with open("sentiment_model.pkl", "rb") as f:
        sentiment_model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return sentiment_model, tfidf

sentiment_model, tfidf = load_sentiment_models()

# -------------------------------------------------------------
# APP HEADER
# -------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise AI Market Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Enterprise AI Market Intelligence & Forecasting Platform")
st.markdown(
    """
    **An end-to-end AI system for market analysis and decision support**

    🔮 Demand Forecasting • 📊 Data Exploration • 💰 Price Optimization • 🗣️ Customer Sentiment Analysis • 📰 Market News Intelligence
    """
)
st.markdown("---")



# -------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------
if "sales_df" not in st.session_state:
    st.session_state.sales_df = None

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer",
    "🔮 Forecasting & Macro-Economics",
    "💰 Price Optimization",
    "🗣️ Customer Reviews",
    "📰 Market News"
])

# =============================================================
# TAB 1: DATA EXPLORER (ADVANCED + USER FRIENDLY)
# =============================================================
with tab1:
    st.header("📊 Data Explorer & Data Health Analysis")

    # ---------- TAB DESCRIPTION ----------
    st.markdown(
        """
        **What does this section do?**  
        This section helps you understand the *quality, behavior, and reliability* of your sales data
        before making predictions.  
        It checks for missing values, unusual spikes, trends, and repeating patterns.
        """
    )

    uploaded = st.file_uploader(
        "Upload Sales CSV (date, sales, promo_ratio optional)",
        type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded)

        # -------- Validation --------
        if not {"date", "sales"}.issubset(df.columns):
            st.error("CSV must contain at least: date, sales")
            st.stop()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        st.session_state.sales_df = df

    if st.session_state.sales_df is not None:
        df = st.session_state.sales_df.copy()

        # -------- KPIs --------
        st.subheader("📌 Dataset Overview")
        st.caption("A quick summary of your uploaded data.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(df))
        c2.metric("Start Date", df["date"].min().strftime("%Y-%m-%d"))
        c3.metric("End Date", df["date"].max().strftime("%Y-%m-%d"))
        c4.metric("Missing Values", int(df.isnull().sum().sum()))
        # -------- Confidence Explanation (Data Quality) --------
        st.subheader("🔍 Data Confidence Level")
        st.caption(
            "This explains how reliable your data is for forecasting, "
            "based on completeness and stability."
        )

        confidence_score = 100

        if df.isnull().sum().sum() > 0:
            confidence_score -= 20
        if avg_volatility := df["sales"].std() > df["sales"].mean() * 0.3:
            confidence_score -= 20

        if confidence_score >= 80:
            st.success(
                f"✅ High confidence ({confidence_score}%) — "
                "Your data is clean and stable enough for accurate forecasting."
            )
        elif confidence_score >= 60:
            st.info(
                f"ℹ️ Medium confidence ({confidence_score}%) — "
                "Forecasts should be interpreted with some caution."
            )
        else:
            st.warning(
                f"⚠️ Low confidence ({confidence_score}%) — "
                "Consider improving data quality before relying on predictions."
            )

        # -------- Suggestions (Data Quality) --------
        if df.isnull().sum().sum() > 0:
            st.warning(
                "⚠️ Your dataset contains missing values. "
                "Consider cleaning or filling them before forecasting to improve accuracy."
            )
        else:
            st.success("✅ No missing values detected. Data quality looks good.")

        # -------- Rolling Statistics --------
        st.subheader("📈 Trend & Volatility Analysis")
        st.caption(
            "This shows how sales change over time. "
            "The rolling line smooths short-term noise, and the shaded area shows variability."
        )

        window = st.slider("Rolling Window (Months)", 3, 12, 6)

        df["rolling_mean"] = df["sales"].rolling(window).mean()
        df["rolling_std"] = df["sales"].rolling(window).std()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["date"], df["sales"], label="Actual Sales", alpha=0.5)
        ax.plot(df["date"], df["rolling_mean"], label="Smoothed Trend", linewidth=2)
        ax.fill_between(
            df["date"],
            df["rolling_mean"] - df["rolling_std"],
            df["rolling_mean"] + df["rolling_std"],
            alpha=0.2,
            label="Sales Variability"
        )
        ax.legend()
        ax.set_title("Sales Trend & Volatility")
        st.pyplot(fig)

        # -------- Volatility Insight --------
        avg_volatility = df["rolling_std"].mean()
        if avg_volatility > df["sales"].mean() * 0.3:
            st.warning(
                "⚠️ High sales volatility detected. "
                "Predictions may have wider uncertainty."
            )
        else:
            st.info(
                "ℹ️ Sales volatility is within a reasonable range."
            )

        # -------- Anomaly Detection --------
        st.subheader("⚠️ Unusual Sales Detection (Anomalies)")
        st.caption(
            "This identifies unusual spikes or drops in sales that do not follow the normal pattern. "
            "They may indicate special events, errors, or one-time incidents."
        )

        contamination = st.slider("Anomaly Sensitivity", 0.01, 0.1, 0.05)

        iso = IsolationForest(contamination=contamination, random_state=42)
        df["anomaly"] = iso.fit_predict(df[["sales"]])

        anomalies = df[df["anomaly"] == -1]

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(df["date"], df["sales"], label="Normal Sales")
        ax2.scatter(
            anomalies["date"],
            anomalies["sales"],
            color="red",
            s=80,
            label="Unusual Points"
        )
        ax2.legend()
        ax2.set_title("Detected Unusual Sales Points")
        st.pyplot(fig2)

        # -------- Anomaly Insight --------
        if len(anomalies) > 0:
            st.warning(
                f"⚠️ {len(anomalies)} unusual data points found. "
                "Review them to check for special promotions, data errors, or external events."
            )
        else:
            st.success("✅ No unusual sales behavior detected.")
        # -------- Recommended Actions (Based on Anomalies) --------
        st.subheader("✅ Recommended Actions")
        st.caption(
            "Suggested next steps based on the patterns detected in your data."
        )

        if len(anomalies) > 0:
            st.warning(
                "• Investigate unusual sales points for special promotions, stockouts, or data errors.\n"
                "• Exclude one-time events if they are not expected to repeat.\n"
                "• Add notes or tags to anomaly periods for future reference."
            )
        else:
            st.success(
                "• Sales behavior is consistent.\n"
                "• Data is suitable for forecasting and optimization.\n"
                "• Proceed confidently to demand forecasting."
            )


        # -------- Seasonal Decomposition --------
        st.subheader("📉 Seasonal Pattern Analysis")
        st.caption(
            "This breaks sales into three parts: trend (long-term growth), "
            "seasonality (repeating patterns), and noise."
        )

        if len(df) >= 24:
            ts = df.set_index("date")["sales"]
            decomp = sm.tsa.seasonal_decompose(ts, model="additive", period=12)
            fig3 = decomp.plot()
            fig3.set_size_inches(10, 8)
            st.pyplot(fig3)

            st.info(
                "ℹ️ If seasonal patterns are strong, the forecasting model will "
                "use them to improve future predictions."
            )
        else:
            st.info("Need at least 24 records to clearly detect seasonal patterns.")
        # -------- Confidence Explanation (Seasonality) --------
        st.subheader("📊 Seasonality Confidence")
        st.caption(
            "Indicates how useful seasonal patterns are likely to be for forecasting."
        )

        if len(df) >= 24:
            st.success(
                "✅ Sufficient historical data detected. "
                "Seasonal patterns can be learned reliably by the forecasting model."
            )
        else:
            st.info(
                "ℹ️ Limited historical data. "
                "Seasonality may be weaker, and forecasts may rely more on trends."
            )


        # -------- Download --------
        st.subheader("⬇️ Export Cleaned Data")
        st.caption(
            "Download the processed dataset with calculated metrics for reporting or reuse."
        )

        st.download_button(
            "Download Processed Data",
            df.to_csv(index=False),
            "processed_sales_data.csv"
        )


# =============================================================
# TAB 2: FORECASTING & MACRO-ECONOMICS (USER FRIENDLY)
# =============================================================
with tab2:
    st.header("🔮 Forecasting with Macro-Economic Context")

    # ---------- TAB DESCRIPTION ----------
    st.markdown(
        """
        **What does this section do?**  
        This section predicts future sales based on your historical data.
        It also allows you to simulate the effect of promotions and compare
        your business trend with the overall market.
        """
    )

    if st.session_state.sales_df is None:
        st.warning("Upload data in Data Explorer first.")
        st.stop()

    df = st.session_state.sales_df.copy()

    # ---------- USER CONTROLS ----------
    st.subheader("🎛️ Forecast Settings")
    st.caption(
        "Adjust these settings to explore different future scenarios."
    )

    c1, c2, c3 = st.columns(3)
    horizon = c1.slider(
        "Forecast Horizon (Months)",
        1, 24, 12,
        help="How many months into the future you want predictions for."
    )
    planned_promo = c2.slider(
        "Future Promo Ratio",
        0.0, 1.0, 0.2,
        help="Expected average promotion level in the future."
    )
    overlay_macro = c3.checkbox(
        "Overlay S&P 500 Index",
        help="Compare your sales trend with overall market performance."
    )

    # ---------- PREPARE DATA ----------
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    prophet_df["promo_ratio"] = df["promo_ratio"] if "promo_ratio" in df.columns else 0.0

    # ---------- TRAIN MODEL ----------
    with st.spinner("Training forecasting model on your data..."):
        model = Prophet()
        model.add_regressor("promo_ratio")
        model.fit(prophet_df)

    # ---------- CREATE FUTURE ----------
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    future["promo_ratio"] = list(prophet_df["promo_ratio"]) + [planned_promo] * horizon

    forecast = model.predict(future)

    # ---------- KEY RESULTS ----------
    st.subheader("📊 Forecast Summary")
    st.caption(
        "Key indicators derived from the predicted future sales."
    )

    future_forecast = forecast.tail(horizon)
    next_month = future_forecast["yhat"].iloc[0]

    growth_rate = (
        (future_forecast["yhat"].mean() - df["sales"].mean())
        / df["sales"].mean()
        * 100
    )

    uncertainty = (future_forecast["yhat_upper"] - future_forecast["yhat_lower"]).mean()

    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Next Month Forecast",
        f"{next_month:,.0f}",
        help="Expected sales for the immediate next month."
    )
    k2.metric(
        "Expected Growth",
        f"{growth_rate:.2f}%",
        help="Average growth compared to historical sales."
    )
    k3.metric(
        "Forecast Uncertainty",
        f"{uncertainty:,.0f}",
        help="Higher value means predictions have more variability."
    )
        # ---------- FORECAST CONFIDENCE ----------
    st.subheader("🔍 Forecast Confidence")
    st.caption(
        "This explains how reliable the forecast is based on data history and uncertainty."
    )

    confidence_score = 100

    if len(df) < 24:
        confidence_score -= 20
    if uncertainty > df["sales"].mean() * 0.4:
        confidence_score -= 30

    if confidence_score >= 80:
        st.success(
            f"✅ High confidence ({confidence_score}%) — "
            "Forecast is based on sufficient data with manageable uncertainty."
        )
    elif confidence_score >= 60:
        st.info(
            f"ℹ️ Medium confidence ({confidence_score}%) — "
            "Forecast is usable, but decisions should include a safety buffer."
        )
    else:
        st.warning(
            f"⚠️ Low confidence ({confidence_score}%) — "
            "Forecast uncertainty is high. Use predictions cautiously."
        )


    # ---------- FORECAST VISUALIZATION ----------
    st.subheader("📈 Forecast Visualization")
    st.caption(
        "The shaded area represents uncertainty. Wider bands indicate higher risk."
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["date"], df["sales"], label="Historical Sales", color="black")
    ax1.plot(forecast["ds"], forecast["yhat"], "--", label="Predicted Sales")
    ax1.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2
    )
    ax1.set_ylabel("Sales")

    # ---------- MACRO OVERLAY ----------
    if overlay_macro:
        try:
            spy = yf.download("^GSPC", start=df["date"].min(), progress=False)
            spy_m = spy["Close"].resample("MS").mean()

            ax2 = ax1.twinx()
            ax2.plot(
                spy_m.index,
                spy_m,
                color="green",
                alpha=0.4,
                label="S&P 500 Index"
            )
            ax2.set_ylabel("Market Index")
        except:
            st.warning("Unable to fetch market index data.")

    ax1.legend(loc="upper left")
    ax1.set_title("Sales Forecast with Market Comparison")
    st.pyplot(fig)

    # ---------- BUSINESS INSIGHTS ----------
    st.subheader("🧠 What does this mean for your business?")
    if growth_rate > 10:
        st.success(
            "📈 Strong growth expected. Consider increasing inventory, "
            "marketing spend, or expanding operations."
        )
    elif growth_rate < -5:
        st.warning(
            "📉 Potential slowdown detected. You may want to control costs "
            "or reduce inventory risk."
        )
    else:
        st.info(
            "➖ Sales are expected to remain stable. Maintain current strategy."
        )
    # ---------- RECOMMENDED ACTIONS ----------
    st.subheader("✅ Recommended Actions")
    st.caption(
        "Suggested next steps based on forecasted growth and risk level."
    )

    if growth_rate > 10 and confidence_score >= 70:
        st.success(
            "• Increase inventory and production planning.\n"
            "• Strengthen marketing campaigns to capture demand.\n"
            "• Prepare operational capacity for growth."
        )
    elif growth_rate < -5:
        st.warning(
            "• Reduce excess inventory risk.\n"
            "• Focus on cost optimization.\n"
            "• Monitor demand closely over the next few months."
        )
    else:
        st.info(
            "• Maintain current business strategy.\n"
            "• Monitor trends regularly.\n"
            "• Use promotional strategies selectively."
        )
    st.info(
        "📌 **Note:** Forecasts are estimates based on historical patterns. "
        "Unexpected events (market shocks, policy changes, supply issues) "
        "may impact actual results."
    )

    # ---------- PDF EXPORT ----------
    st.subheader("📄 Download Forecast Report")
    st.caption(
        "Generate a PDF summary of this forecast for sharing or reporting."
    )

    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, "AI Market Forecast Report", ln=True, align="C")

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)

    pdf.cell(0, 10, f"Records: {len(df)}", ln=True)
    pdf.cell(0, 10, f"Forecast Horizon: {horizon} months", ln=True)
    pdf.cell(0, 10, f"Planned Promo Ratio: {planned_promo}", ln=True)
    pdf.cell(0, 10, f"Next Month Forecast: {next_month:,.0f}", ln=True)

    tmp = os.path.join(tempfile.gettempdir(), "forecast_tab2.png")
    fig.savefig(tmp)
    pdf.image(tmp, w=190)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "⬇️ Download Forecast PDF",
        pdf_bytes,
        "forecast_report.pdf",
        "application/pdf"
    )

# =============================================================
# TAB 3: PRICE OPTIMIZATION (USER FRIENDLY + ADVANCED)
# =============================================================
with tab3:
    st.header("💰 Price & Promotion Optimization")

    # ---------- TAB DESCRIPTION ----------
    st.markdown(
        """
        **What does this section do?**  
        This section helps you decide *how much discount or promotion to offer* 
        in order to **maximize profit**, not just sales.
        It simulates customer behavior under different discount levels.
        """
    )

    if st.session_state.sales_df is None:
        st.warning("Please upload sales data in Data Explorer first.")
        st.stop()

    df = st.session_state.sales_df.copy()
    base_sales = df["sales"].mean()

    # ---------- ASSUMPTIONS ----------
    st.subheader("📌 Optimization Assumptions")
    st.caption(
        "Adjust these values to reflect how your customers usually respond to discounts."
    )

    c1, c2, c3 = st.columns(3)
    elasticity = c1.slider(
        "Demand Elasticity",
        0.5, 3.0, 1.5,
        help="How strongly customers respond to discounts. Higher value = more price-sensitive customers."
    )
    margin = c2.slider(
        "Profit Margin (%)",
        5, 80, 30,
        help="Percentage of revenue that is profit after costs."
    )
    max_discount = c3.slider(
        "Maximum Discount (%)",
        5, 70, 50,
        help="Highest discount you are willing to consider."
    )

    # ---------- MODEL ----------
    promo_range = np.linspace(0, max_discount / 100, 50)

    demand = base_sales * (1 + elasticity * promo_range)
    revenue = demand * (1 - promo_range)
    profit = revenue * (margin / 100)

    optimal_idx = np.argmax(profit)
    optimal_discount = promo_range[optimal_idx]
    optimal_profit = profit[optimal_idx]

    # ---------- VISUALIZATION ----------
    st.subheader("📈 Profit vs Discount Analysis")
    st.caption(
        "This curve shows how profit changes as discount increases. "
        "The green line marks the best balance between volume and margin."
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(promo_range * 100, profit, label="Projected Profit", color="purple")
    ax.axvline(
        optimal_discount * 100,
        linestyle="--",
        color="green",
        label=f"Optimal Discount: {optimal_discount:.0%}"
    )
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Profit")
    ax.set_title("Profit Optimization Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ---------- METRICS ----------
    st.subheader("📊 Optimization Results")
    m1, m2 = st.columns(2)
    m1.metric(
        "Recommended Discount",
        f"{optimal_discount:.1%}",
        help="Discount level that gives the highest projected profit."
    )
    m2.metric(
        "Projected Monthly Profit",
        f"{optimal_profit:,.0f}",
        help="Estimated profit after applying the recommended discount."
    )
    # ---------- OPTIMIZATION CONFIDENCE ----------
    st.subheader("🔍 Optimization Confidence")
    st.caption(
        "This indicates how reliable the recommended discount is based on customer behavior assumptions."
    )

    confidence_score = 100

    # Confidence heuristics (no model change)
    if elasticity < 0.8 or elasticity > 2.5:
        confidence_score -= 20
    if (profit.max() - profit.min()) < base_sales * 0.05:
        confidence_score -= 20

    if confidence_score >= 80:
        st.success(
            f"✅ High confidence ({confidence_score}%) — "
            "The profit curve shows a clear optimal discount level."
        )
    elif confidence_score >= 60:
        st.info(
            f"ℹ️ Medium confidence ({confidence_score}%) — "
            "The recommendation is useful but should be tested gradually."
        )
    else:
        st.warning(
            f"⚠️ Low confidence ({confidence_score}%) — "
            "Customer response to discounts is uncertain. Proceed cautiously."
        )


    # ---------- BUSINESS RECOMMENDATION ----------
    st.subheader("🧠 What should you do?")
    if elasticity > 2:
        st.success(
            f"Customers are highly price-sensitive. "
            f"A **{optimal_discount:.0%} discount** can significantly boost demand "
            f"while still maximizing profit."
        )
    elif elasticity < 1:
        st.info(
            f"Customers are less sensitive to price. "
            f"Heavy discounts are not necessary; "
            f"the model suggests **{optimal_discount:.0%}** as a safe option."
        )
    else:
        st.success(
            f"A balanced discount of **{optimal_discount:.0%}** is recommended "
            f"to optimize profit without hurting margins."
        )
    # ---------- RECOMMENDED ACTIONS ----------
    st.subheader("✅ Recommended Actions")
    st.caption(
        "Practical next steps based on the optimization results."
    )

    if confidence_score >= 80:
        st.success(
            "• Apply the recommended discount in upcoming campaigns.\n"
            "• Monitor sales and profit closely for the next 1–2 cycles.\n"
            "• Use this discount as a benchmark for future pricing strategies."
        )
    elif confidence_score >= 60:
        st.info(
            "• Start with a small pilot promotion.\n"
            "• Compare results against historical performance.\n"
            "• Adjust discount levels incrementally based on response."
        )
    else:
        st.warning(
            "• Avoid aggressive discounts initially.\n"
            "• Run A/B tests or short-term trials.\n"
            "• Collect more data to refine customer sensitivity estimates."
        )


# =============================================================
# TAB 4: ANALYSING CUSTOMER REVIEWS (USER FRIENDLY)
# =============================================================
with tab4:
    st.header("🗣️ Customer Review Sentiment Analysis")

    # ---------- TAB DESCRIPTION ----------
    st.markdown(
        """
        **What does this section do?**  
        This section analyzes customer feedback and tells you whether reviews are
        **Positive**, **Neutral**, or **Negative**.  
        It helps you understand customer satisfaction and identify areas for improvement.
        """
    )

    if sentiment_model is None or tfidf is None:
        st.error("Sentiment model not loaded. Please check .pkl files.")
        st.stop()

    tab_single, tab_batch = st.tabs(["✍️ Single Review", "📁 Bulk CSV Analysis"])

    # ---------------- SINGLE REVIEW ----------------
    with tab_single:
        st.subheader("✍️ Analyze a Single Review")
        st.caption(
            "Enter one customer review to instantly understand the customer's sentiment."
        )

        review = st.text_area(
            "Customer Feedback:",
            height=120,
            placeholder="The product quality is excellent but delivery was slow."
        )

        if st.button("Analyze Review"):
            if not review.strip():
                st.warning("Please enter some text.")
            else:
                vec = tfidf.transform([review])
                pred = sentiment_model.predict(vec)[0]
                res = str(pred).upper()

                st.subheader("📊 Sentiment Result")
                if "POS" in res:
                    st.success(
                        "😊 **Positive** — Customers are happy with this experience."
                    )
                elif "NEG" in res:
                    st.error(
                        "😞 **Negative** — Indicates dissatisfaction. Consider investigating."
                    )
                else:
                    st.warning(
                        "😐 **Neutral** — Mixed or balanced feedback."
                    )
                # ---------- CONFIDENCE EXPLANATION ----------
                st.subheader("🔍 Confidence Explanation")
                st.caption(
                    "This indicates how strongly the model detected the sentiment from the text."
                )

                text_length = len(review.split())

                if text_length >= 8:
                    st.success(
                        "✅ High confidence — The review contains enough context "
                        "for reliable sentiment detection."
                    )
                elif text_length >= 4:
                    st.info(
                        "ℹ️ Medium confidence — Sentiment detected, "
                        "but additional context could improve reliability."
                    )
                else:
                    st.warning(
                        "⚠️ Low confidence — Very short feedback may lead "
                        "to ambiguous sentiment interpretation."
                    )

                # ---------- RECOMMENDED ACTION ----------
                st.subheader("✅ Recommended Action")
                if "POS" in res:
                    st.success(
                        "• Continue current product or service approach.\n"
                        "• Highlight this feedback in testimonials or reviews."
                    )
                elif "NEG" in res:
                    st.warning(
                        "• Investigate the issue mentioned by the customer.\n"
                        "• Consider corrective actions or follow-up support."
                    )
                else:
                    st.info(
                        "• Monitor similar feedback trends.\n"
                        "• Gather more detailed customer responses."
                    )


    # ---------------- BULK ANALYSIS ----------------
    with tab_batch:
        st.subheader("📁 Analyze Multiple Reviews")
        st.caption(
            "Upload a CSV file to analyze customer sentiment at scale."
        )
        st.info("CSV must contain a column named: review / text / comment / feedback")

        file = st.file_uploader("Upload Reviews CSV", type=["csv"])

        if file:
            df_rev = pd.read_csv(file)
            col_candidates = [
                c for c in df_rev.columns
                if c.lower() in ["review", "text", "comment", "feedback"]
            ]

            if not col_candidates:
                st.error("No valid text column found.")
                st.stop()

            text_col = col_candidates[0]

            with st.spinner("Running sentiment analysis..."):
                vecs = tfidf.transform(df_rev[text_col].astype(str))
                df_rev["sentiment"] = sentiment_model.predict(vecs)

            # Distribution
            st.subheader("📊 Overall Sentiment Distribution")
            st.caption(
                "This chart shows the overall mood of your customers."
            )

            fig, ax = plt.subplots()
            df_rev["sentiment"].value_counts().plot.pie(
                autopct="%1.1f%%", startangle=90, ax=ax
            )
            ax.set_ylabel("")
            st.pyplot(fig)

            # WordCloud
            positives = " ".join(
                df_rev[
                    df_rev["sentiment"].astype(str).str.contains("POS", case=False)
                ][text_col]
            )

            if positives.strip():
                st.subheader("☁️ Common Words in Positive Reviews")
                st.caption(
                    "Frequently mentioned words in positive feedback."
                )
                wc = WordCloud(
                    width=800, height=300, background_color="white"
                ).generate(positives)
                st.image(wc.to_array())

            st.subheader("🔍 Review Preview")
            st.dataframe(df_rev.head(20))

            st.download_button(
                "⬇️ Download Sentiment Results",
                df_rev.to_csv(index=False),
                "sentiment_results.csv"
            )
            # ---------- OVERALL CONFIDENCE ----------
            st.subheader("🔍 Overall Sentiment Confidence")
            st.caption(
                "Confidence is higher when the majority sentiment is clear and dominant."
            )

            sentiment_counts = df_rev["sentiment"].value_counts(normalize=True)

            dominant_ratio = sentiment_counts.iloc[0]

            if dominant_ratio >= 0.65:
                st.success(
                    "✅ High confidence — A clear dominant customer sentiment is observed."
                )
            elif dominant_ratio >= 0.45:
                st.info(
                    "ℹ️ Medium confidence — Mixed customer opinions detected."
                )
            else:
                st.warning(
                    "⚠️ Low confidence — No clear customer sentiment trend."
                )

            # ---------- RECOMMENDED ACTIONS ----------
            st.subheader("✅ Recommended Actions")
            if "POS" in sentiment_counts.index[0]:
                st.success(
                    "• Strengthen marketing using positive customer voice.\n"
                    "• Maintain product/service quality."
                )
            elif "NEG" in sentiment_counts.index[0]:
                st.warning(
                    "• Identify recurring complaints.\n"
                    "• Prioritize fixes in high-impact areas."
                )
            else:
                st.info(
                    "• Conduct surveys for deeper customer insights.\n"
                    "• Encourage detailed feedback."
                )

# =============================================================
# TAB 5: MARKET NEWS (USER FRIENDLY + INTELLIGENT)
# =============================================================
with tab5:
    st.header("📰 Global Market News Intelligence")

    # ---------- TAB DESCRIPTION ----------
    st.markdown(
        """
        **What does this section do?**  
        This section shows the latest market-related news for a selected company or index.
        News can strongly influence demand, customer sentiment, and sales trends.
        """
    )

    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input(
        "Enter Ticker Symbol",
        "AAPL",
        help="Examples: AAPL, MSFT, TSLA, SPY"
    )
    fetch = c2.button("Fetch News")

    if fetch:
        try:
            with st.spinner("Fetching latest news..."):
                stock = yf.Ticker(ticker)
                news = stock.news

            if not news:
                st.info("No recent news found.")
            else:
                st.subheader("🗞️ Top Recent Articles")
                st.caption(
                    "Review these headlines to understand external factors affecting your business."
                )

                for item in news[:5]:
                    content = item.get("content", {})

                    title = (
                        item.get("title") or
                        content.get("title") or
                        item.get("headline")
                    )

                    link = (
                        item.get("link") or
                        content.get("canonicalUrl", {}).get("url") or
                        content.get("previewUrl")
                    )

                    publisher = (
                        item.get("provider", {}).get("displayName") or
                        content.get("provider", {}).get("displayName") or
                        "Unknown Source"
                    )

                    pub_time = item.get("providerPublishTime")
                    date_str = (
                        pd.to_datetime(pub_time, unit="s").strftime("%Y-%m-%d %H:%M")
                        if pub_time else "Recent"
                    )

                    if not title:
                        continue

                    with st.expander(f"📰 {title}", expanded=True):
                        st.markdown(f"**Source:** {publisher}")
                        st.markdown(f"**Published:** {date_str}")
                        if link:
                            st.markdown(f"[🔗 Read full article]({link})")

                        thumb = content.get("thumbnail", {}).get("resolutions", [{}])[0].get("url")
                        if thumb:
                            st.image(thumb, width=250)
                # ---------- NEWS CONFIDENCE ----------
                st.subheader("🔍 News Impact Confidence")
                st.caption(
                    "Confidence is based on the volume and recency of news coverage."
                )

                article_count = len(news[:5])

                if article_count >= 4:
                    st.success(
                        "✅ High confidence — Multiple recent articles suggest "
                        "strong market relevance."
                    )
                elif article_count >= 2:
                    st.info(
                        "ℹ️ Medium confidence — Limited but useful market signals."
                    )
                else:
                    st.warning(
                        "⚠️ Low confidence — Insufficient news to draw conclusions."
                    )


                # -------- BUSINESS SUGGESTION --------
                st.subheader("🧠 How to use this information")
                st.info(
                    "If multiple news articles show positive developments, "
                    "you may expect higher demand. Negative news may indicate "
                    "potential risks or demand slowdown."
                )
                # ---------- RECOMMENDED ACTIONS ----------
                st.subheader("✅ Recommended Actions")
                st.caption(
                    "How to use market news effectively in business planning."
                )

                st.info(
                    "• Align short-term sales and inventory decisions with market sentiment.\n"
                    "• Combine news insights with demand forecasts for better accuracy.\n"
                    "• Monitor news regularly for sudden market shifts."
                )


        except Exception as e:
            st.error(f"Failed to fetch news: {e}")

