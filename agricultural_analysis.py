# Agricultural Commodity Price Analysis
# Data collection, diagnostics, regressions, and sensitivity analysis


from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from fredapi import Fred
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
FRED_API_KEY = os.environ["FRED_API_KEY"]
USDA_API_KEY = os.getenv("USDA_API_KEY", "")

mysql_config = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.environ["MYSQL_PASSWORD"],
    "port": int(os.getenv("MYSQL_PORT", "3306"))
}
db_name = "agri_monthly_db"

try:
    conn_test = mysql.connector.connect(**mysql_config, database=db_name)
    print("MySQL connection successful.")
    conn_test.close()
except Error as e:
    print("Connection failed:", e)

def write_to_mysql(df, table_name):
    conn = None
    try:
        conn = mysql.connector.connect(**mysql_config, database=db_name)
        cursor = conn.cursor()

        cursor.execute(f"TRUNCATE TABLE {table_name};")

        for _, row in df.iterrows():
            cols = ", ".join(df.columns)
            placeholders = ", ".join(["%s"] * len(df.columns))
            sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(row))
        conn.commit()
        print(f"✅ {table_name} written successfully; rows: {len(df)}")
    except Error as e:
        print("Database write error:", e)
    finally:
        if conn is not None and conn.is_connected():
            cursor.close()
            conn.close()

fred = Fred(api_key=FRED_API_KEY)

series_dict = {
    "WTI_OIL": "MCOILWTICO",
    "FFR": "FEDFUNDS",
    "USD_INDEX": "TWEXB",
    "CPI": "CPIAUCSL",
}

df_fred = pd.DataFrame()
for name, sid in series_dict.items():
    s = fred.get_series(sid, observation_start="1996-01-01")
    df_fred[name] = s
df_fred.index.name = "date"
df_fred.reset_index(inplace=True)

df_fred = df_fred.replace({np.nan: None})

write_to_mysql(df_fred, "macro_fred")

df_price_raw = pd.read_csv(os.getenv("CROP_PRICE_CSV", "crop_price.csv"))

month_map = {
    "JAN":1, "FEB":2, "MAR":3, "APR":4, "MAY":5, "JUN":6,
    "JUL":7, "AUG":8, "SEP":9, "OCT":10, "NOV":11, "DEC":12
}
df_price_raw["month"] = df_price_raw["Period"].map(month_map)

df_price_raw["year"] = df_price_raw["Year"].astype(int)
df_price_raw["date"] = pd.to_datetime(df_price_raw[["year","month"]].assign(day=1))

df_price_raw["price"] = pd.to_numeric(df_price_raw["Value"], errors="coerce")

df_crop_raw = df_price_raw[["date","year","month","Commodity","price"]].rename(columns={"Commodity":"commodity_desc"})

df_crop_raw = df_crop_raw.replace({np.nan: None})

write_to_mysql(df_crop_raw, "crop_price")

def read_sql_df(query):
    conn = None
    try:

        conn = mysql.connector.connect(**mysql_config)

        conn.database = "agri_monthly_db"
        df = pd.read_sql(query, conn)
        return df
    except Error as e:
        print("Database read error:", e)
        return None
    finally:
        if conn is not None and conn.is_connected():
            conn.close()

df_crop = read_sql_df("SELECT * FROM crop_price;")
df_macro = read_sql_df("SELECT * FROM macro_fred;")

df_panel = pd.merge(df_crop, df_macro, on="date", how="left")

df_panel["D_RFS"] = np.where(df_panel["year"] >= 2007, 1, 0)

df_panel["D_drought2012"] = np.where(df_panel["year"] == 2012, 1, 0)

df_panel["D_tradewar"] = np.where((df_panel["year"] >= 2018) & (df_panel["year"] <= 2019), 1, 0)

df_panel["D_covid"] = np.where((df_panel["year"] >= 2020) & (df_panel["year"] <= 2021), 1, 0)

df_panel["D_rus_ukr"] = np.where(df_panel["year"] >= 2022, 1, 0)

print("\n==== Event dummy preview ====")
print(df_panel[["year","D_RFS","D_drought2012","D_tradewar","D_covid","D_rus_ukr"]].head(15))

df_panel = df_panel.sort_values("date", ascending=True).reset_index(drop=True)
print("Sorted by date in ascending order")

print("\n==== Missing value counts ====")
print(df_panel.isnull().sum())

plt.figure(figsize=(12,5))
sns.heatmap(df_panel.isnull(), cbar=False)
plt.title("Missing Value Heatmap")
plt.tight_layout()
plt.show()

crop_list = ["CORN", "SOYBEANS", "WHEAT"]

for crop in crop_list:
    crop_all_df = df_panel[df_panel["commodity_desc"] == crop].copy()
    total_count = len(crop_all_df)
    crop_all_df["has_missing"] = crop_all_df.isna().any(axis=1)

    missing_count = crop_all_df["has_missing"].sum()
    valid_count = crop_all_df.dropna().shape[0]

    all_price = crop_all_df["price"].dropna()
    valid_price = crop_all_df[crop_all_df["has_missing"] == False]["price"]

    mean_p = all_price.mean()
    std_p = all_price.std()
    sigma1_upper = mean_p + std_p
    sigma1_lower = mean_p - std_p
    sigma2_upper = mean_p + 2*std_p
    sigma2_lower = mean_p - 2*std_p

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{crop} Price | Total={total_count}, Rows with any missing={missing_count}, Valid={valid_count}", fontsize=12)

    sns.histplot(all_price, kde=True, ax=ax1, color="#4472C4", alpha=0.6, label="All price observations (price non-null)")
    sns.histplot(valid_price, kde=True, ax=ax1, color="#2E8B57", alpha=0.7, label="Valid for regression (all variables non-null)")

    ax1.axvline(mean_p, color="red", linestyle="-", lw=2, label="Mean (all price)")
    ax1.axvline(sigma1_upper, color="orange", linestyle="--", label="Mean ±1σ")
    ax1.axvline(sigma1_lower, color="orange", linestyle="--")
    ax1.axvline(sigma2_upper, color="green", linestyle=":", label="Mean ±2σ")
    ax1.axvline(sigma2_lower, color="green", linestyle=":")
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Frequency")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Histogram")

    ax1.text(0.03, 0.95, f"Missing rows (macro NaN): {missing_count}", transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", facecolor="red", alpha=0.2))

    box_data = [all_price, valid_price]
    box = ax2.boxplot(box_data, patch_artist=True)

    ax2.set_xticklabels(["All price", "Valid for regression"])
    colors = ["#5B9BD5", "#2E8B57"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax2.axvline(mean_p, color="red", linestyle="-", lw=2)
    ax2.axvline(sigma1_upper, color="orange", linestyle="--")
    ax2.axvline(sigma1_lower, color="orange", linestyle="--")
    ax2.axvline(sigma2_upper, color="green", linestyle=":")
    ax2.axvline(sigma2_lower, color="green", linestyle=":")
    ax2.set_ylabel("Price")
    ax2.set_title("Boxplot")

    plt.tight_layout()
    plt.show()

    print(f"\n==== {crop} ====")
    print(f"Total observations: {total_count}, Rows with missing macro data: {missing_count}, Complete-case observations: {valid_count}")
    print(f"Mean price = {mean_p:.2f}, σ = {std_p:.2f}")

X_cols = ["WTI_OIL","FFR","USD_INDEX","CPI","D_RFS","D_drought2012","D_tradewar","D_covid","D_rus_ukr"]

df_panel["has_missing_x"] = df_panel[X_cols].isnull().any(axis=1)

X_raw = df_panel[X_cols]
y_raw = df_panel["price"]

imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X_raw)
X_imp_df = pd.DataFrame(X_imputed, columns=X_cols)

for idx, col in enumerate(X_cols):
    df_panel[col] = X_imp_df.iloc[:, idx]

for crop in crop_list:
    print ("\n" + "="*50)
print (f"========== {crop} Crop regression ==========")
sub_df = df_panel [df_panel ["commodity_desc"] == crop].copy ()
y_sub = sub_df ["price"]
X_sub = sub_df [X_cols]
X_sub = sm.add_constant (X_sub)
mod = sm.OLS (y_sub, X_sub)

res_sub = mod.fit(cov_type="HAC", cov_kwds={"maxlags":12})
print(res_sub.summary())

result_dict = {}

X_sm_all = sm.add_constant(X_imp_df)
mod_all = sm.OLS(y_raw, X_sm_all)
res_all = mod_all.fit(cov_type="HAC", cov_kwds={"maxlags":12})
result_dict["Full Panel"] = res_all

for crop in crop_list:
    sub_df = df_panel[df_panel["commodity_desc"] == crop].copy()
    y_sub = sub_df["price"]
    X_sub = sub_df[X_cols]
    X_sub = sm.add_constant(X_sub)
    mod = sm.OLS(y_sub, X_sub)
    res_sub = mod.fit(cov_type="HAC", cov_kwds={"maxlags":12})
    result_dict[crop] = res_sub

coef_data = []
for model_name, res in result_dict.items():
    df_coef = pd.DataFrame({
        "model": model_name,
        "var_name": res.params.index,
        "coef": res.params.values,
        "lower95": res.conf_int()[0].values,
        "upper95": res.conf_int()[1].values
    })
    coef_data.append(df_coef)
coef_df = pd.concat(coef_data)

coef_df = coef_df[coef_df["var_name"] != "const"]

var_names = coef_df["var_name"].unique()
fig, ax = plt.subplots(figsize=(10,7))
colors = ["black","#1f77b4","#ff7f0e","#2ca02c"]
model_order = ["Full Panel","CORN","SOYBEANS","WHEAT"]
offset = [-0.3, -0.1, 0.1, 0.3]

for i, model in enumerate(model_order):
    sub = coef_df[coef_df["model"] == model]

    y_pos = np.array([list(var_names).index(v) for v in sub["var_name"]])

    err_low = np.array(sub["coef"] - sub["lower95"])
    err_high = np.array(sub["upper95"] - sub["coef"])
    ax.errorbar(x=sub["coef"],
                y=y_pos + offset[i],
                xerr=[err_low, err_high],
                fmt="o", capsize=4, color=colors[i], label=model)

ax.axvline(x=0, linestyle="--", color="red", alpha=0.7)
ax.set_yticks(range(len(var_names)))
ax.set_yticklabels(var_names)
ax.set_xlabel("Coefficient Estimate with 95% Confidence Interval")
ax.set_title("Coefficient Forest Plot: Full Sample and Crop‑Specific Regressions")
ax.legend()
plt.tight_layout()
plt.show()

df_panel["season_dummy"] = np.where(df_panel["month"].isin([7,8,9,10]), 1, 0)

indpro_series = fred.get_series("INDPRO", observation_start="1996-01-01")
df_indpro = indpro_series.reset_index()
df_indpro.columns = ["date", "INDPRO"]

df_indpro["ymd"] = df_indpro["date"].dt.strftime("%Y-%m-%d")

df_panel["ymd"] = pd.to_datetime(df_panel["date"], errors="coerce").dt.strftime("%Y-%m-%d")

df_panel = df_panel.reset_index(drop=True)

df_panel = pd.merge(
    df_panel,
    df_indpro[["ymd","INDPRO"]],
    on="ymd",
    how="left"
)

df_panel["INDPRO_yoy"] = df_panel["INDPRO"].pct_change(12) * 100
df_panel["CPI_yoy"] = df_panel["CPI"].pct_change(12) * 100

def assign_mlin_cycle(gdp_yoy, cpi_yoy):
    """
    Recovery: output up, inflation down
    Overheating: output up, inflation up
    Stagflation: output down, inflation up
    Recession: output down, inflation down
    """
    if np.isnan(gdp_yoy) or np.isnan(cpi_yoy):
        return np.nan
    gdp_up = gdp_yoy > 0
    cpi_up = cpi_yoy > 0
    if gdp_up and not cpi_up:
        return "Recovery"
    elif gdp_up and cpi_up:
        return "Overheat"
    elif not gdp_up and cpi_up:
        return "Stagflation"
    else:
        return "Recession"

df_panel["cycle"] = df_panel.apply(lambda r: assign_mlin_cycle(r["INDPRO_yoy"], r["CPI_yoy"]), axis=1)

print("\n" + "="*70)
print("===== Merrill Clock regime counts =====")
print(df_panel["cycle"].value_counts(dropna=False))

drop_cols = [c for c in df_panel.columns if c.startswith("cycle_") and c!="cycle"]
df_panel.drop(columns=drop_cols, errors="ignore", inplace=True)
cycle_dummies = pd.get_dummies(df_panel["cycle"], prefix="cycle", drop_first=True).reset_index(drop=True)
df_panel = pd.concat([df_panel.reset_index(drop=True), cycle_dummies], axis=1)
for col in ["cycle_Recovery","cycle_Overheat","cycle_Stagflation"]:
    if col not in df_panel.columns:
        df_panel[col]=0

df_panel["WTI_DROUGHT"] = df_panel["WTI_OIL"] * df_panel["D_drought2012"]

df_panel["FFR_SEASON"] = df_panel["FFR"] * df_panel["season_dummy"]

X_inter_full = X_cols + ["WTI_DROUGHT", "FFR_SEASON"]

for crop in crop_list:
    print("\n" + "="*80)
    print(f"######################## {crop} Analysis ########################")
    try:
        sub_df = df_panel[df_panel["commodity_desc"] == crop].copy()
        sub_df = sub_df.dropna(subset=["cycle"])
        sub_df = sub_df.sort_values("date").reset_index(drop=True)

        if len(sub_df) < 15:
            print(f"Warning: {crop} valid sample size {len(sub_df)} too small, skip this crop.")
            continue

        y = sub_df["price"]
        X = sub_df[X_inter_full]
        X_const = sm.add_constant(X)
        res_model = sm.OLS(y, X_const).fit(cov_type="HAC", cov_kwds={"maxlags":12})
        print(res_model.summary())

        resid = res_model.resid

        lb_test = acorr_ljungbox(resid, lags=[1,6,12,24], return_df=True)
        print(f"\n【{crop}｜Residual Ljung‑Box White‑Noise Test】")
        print(lb_test)

        lowess_out = lowess(endog=y, exog=np.arange(len(y)), frac=0.15)
        trend = lowess_out[:,1]

        plt.figure(figsize=(14,5))
        plt.plot(sub_df["date"], y, label="Original monthly price", alpha=0.6)
        plt.plot(sub_df["date"], trend, color="red", lw=2.2, label="Lowess long‑term trend (frac=0.15)")
        plt.title(f"{crop}: Original Price vs Lowess Long‑Term Trend")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(f"{crop}_Lowess.png", dpi=300)
        plt.close()

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,4))
        sm.graphics.tsa.plot_acf(y, lags=36, ax=ax1)
        ax1.set_title(f"{crop}: ACF of Original Price Series")
        sm.graphics.tsa.plot_acf(resid, lags=36, ax=ax2)
        ax2.set_title(f"{crop}: ACF of Model Residuals")
        plt.tight_layout()
        plt.savefig(f"{crop}_ACF.png", dpi=300)
        plt.close()

        print(f"======== {crop} Analysis Finished ========\n")

    except Exception as e:
        print(f"!!!! Error occurred when processing {crop} : {str(e)}")
        continue

result_store = []
lb_test_store = []

cycle_list = df_panel["cycle"].dropna().unique()

for crop in crop_list:
    print("\n" + "="*75)
    print(f"========== {crop} | Regime regressions with interactions ==========")
    crop_data = df_panel[df_panel["commodity_desc"] == crop].copy()

    for cyc in cycle_list:
        sub_data = crop_data[crop_data["cycle"] == cyc].copy()

        y = sub_data["price"]
        X = sub_data[X_inter_full]
        X = sm.add_constant(X)

        res_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags":12})

        print(f"\n---------- {crop} | {cyc} N={len(sub_data)} ----------")
        print(res_model.summary())

        resid = res_model.resid
        n_obs = len(resid)

        max_lag_allowed = n_obs // 2
        test_lags = [lag for lag in [1,6,12,24] if lag <= max_lag_allowed]

        if len(test_lags) >0:
            lb_result = acorr_ljungbox(resid, lags=test_lags, return_df=True)
            print(f"\n==== Ljung-Box White Noise Test | {crop} - {cyc} ====")
            print(lb_result)
            for row in lb_result.itertuples():
                lb_test_store.append({
                    "crop":crop, "cycle":cyc, "obs":n_obs,
                    "lag":row.Index, "lb_stat":row.lb_stat, "lb_pval":row.lb_pvalue
                })
        else:
            print(f"\n==== {crop}-{cyc}:Insufficient sample size(n={n_obs})，cannot run Ljung-Box test ====")

        temp_df = pd.DataFrame({
            "crop": crop,
            "cycle": cyc,
            "variable": res_model.params.index,
            "coef": res_model.params.values,
            "pval": res_model.pvalues.values,
            "r_squared": res_model.rsquared,
            "obs": len(sub_data)
        })
        result_store.append(temp_df)

df_reg_result = pd.concat(result_store, ignore_index=True)
df_lb_all = pd.DataFrame(lb_test_store)

with pd.ExcelWriter("all_crop_cycle_reg_LB_result.xlsx") as writer:
    df_reg_result.to_excel(writer, sheet_name="regression_coef", index=False)
    df_lb_all.to_excel(writer, sheet_name="LjungBox_test", index=False)

print("\n===== Regression and Ljung-Box results saved for all crops and regimes. =====")

window_sizes = [48, 60, 72]
step = 1

trace_vars = ["WTI_OIL", "FFR", "USD_INDEX", "CPI",
              "D_rus_ukr", "D_covid", "WTI_DROUGHT", "FFR_SEASON"]

all_rolling = []

for wsize in window_sizes:
    for crop in crop_list:
        df_crop = df_panel[df_panel["commodity_desc"] == crop].copy()
        df_crop = df_crop.sort_values("date").reset_index(drop=True)
        n_total = len(df_crop)

        for start in range(0, n_total - wsize, step):
            end = start + wsize
            sw = df_crop.iloc[start:end].copy()
            mid_date = sw["date"].iloc[wsize // 2]

            y = sw["price"]
            X = sw[X_inter_full]
            X = sm.add_constant(X)

            try:
                res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
                for var in res.params.index:
                    all_rolling.append({
                        "window": wsize,
                        "crop": crop,
                        "mid_date": mid_date,
                        "variable": var,
                        "coef": res.params[var],
                        "pval": res.pvalues[var],
                        "r2": res.rsquared
                    })
            except Exception as e:

                continue

df_rolling = pd.DataFrame(all_rolling)
df_rolling.to_excel("rolling_window_all_sizes.xlsx", index=False)
print("Rolling regressions saved to rolling_window_all_sizes.xlsx")

for crop in crop_list:
    for var in ["WTI_DROUGHT"]:
        plt.figure(figsize=(13, 5))
        for wsize in window_sizes:
            sub = df_rolling[(df_rolling["crop"] == crop) &
                             (df_rolling["window"] == wsize) &
                             (df_rolling["variable"] == var)]
            sub = sub.sort_values("mid_date")
            plt.plot(pd.to_datetime(sub["mid_date"]), sub["coef"],
                     label=f"window={wsize} months")
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        plt.title(f"{crop} | {var} | Robustness across window sizes")
        plt.xlabel("Window Mid-Date")
        plt.ylabel("Coefficient")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{crop}_{var}_window_robustness.png", dpi=300)
        plt.close()
print("Window robustness plots saved.")

cycle_color = {
    "Recovery": "#e2e2e2",
    "Overheat": "#ffcccc",
    "Stagflation": "#ffddbb",
    "Recession": "#cce5ff"
}

legend_elements = [
    Patch(facecolor=cycle_color["Overheat"], alpha=0.22, label="Overheat"),
    Patch(facecolor=cycle_color["Stagflation"], alpha=0.22, label="Stagflation"),
    Patch(facecolor=cycle_color["Recession"], alpha=0.22, label="Recession"),
    Patch(facecolor=cycle_color["Recovery"], alpha=0.22, label="Recovery")
]

for crop in crop_list:
    sub60 = df_rolling[(df_rolling["crop"] == crop) & (df_rolling["window"] == 60)]
    fig, axes = plt.subplots(nrows=len(trace_vars), ncols=1, figsize=(13, 18), sharex=True)
    temp_crop = df_panel[df_panel["commodity_desc"] == crop].copy()
    temp_crop = temp_crop.sort_values("date").reset_index(drop=True)

    for i, var in enumerate(trace_vars):
        ax = axes[i]
        d = sub60[sub60["variable"] == var].sort_values("mid_date")
        ax.plot(pd.to_datetime(d["mid_date"]), d["coef"], color="#203864", lw=1.3)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)

        for _, r in temp_crop.iterrows():
            dt = pd.to_datetime(r["date"])
            if pd.notna(r["cycle"]):
                ax.axvspan(dt, dt + pd.Timedelta(days=31), alpha=0.22, color=cycle_color.get(r["cycle"], "white"))

        ax.set_ylabel(var, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8.5, frameon=True)
    axes[0].set_zorder(10)

    plt.suptitle(f"{crop} | Rolling‑window coefficients (window=60) with Merrill‑Clock regimes", y=0.995)
    plt.xlabel("Date")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(f"{crop}_rolling_window_60m.png", dpi=300, bbox_inches="tight")
    plt.close()
print("Rolling coefficient plots generated, legend added.")

shock_time_dict = {
    "D_RFS":          "2007-01-01",
    "D_drought2012":  "2012-07-01",
    "D_tradewar":     "2018-01-01",
    "D_covid":        "2020-03-01",
    "D_rus_ukr":      "2022-02-01"
}

var_list = X_inter_full

shock_compare_result = []

for shock_name, shock_date in shock_time_dict.items():
    shock_dt = pd.to_datetime(shock_date)

    for crop in crop_list:

        df_roll = df_rolling[(df_rolling["crop"] == crop) & (df_rolling["window"] == 60)].copy()
        df_roll["mid_date"] = pd.to_datetime(df_roll["mid_date"])

        df_before = df_roll[df_roll["mid_date"] < shock_dt].copy()
        df_after  = df_roll[df_roll["mid_date"] >= shock_dt].copy()

        for var in var_list:

            sub_b = df_before[df_before["variable"] == var]
            mean_b = sub_b["coef"].mean() if len(sub_b) > 0 else np.nan

            sub_a = df_after[df_after["variable"] == var]
            mean_a = sub_a["coef"].mean() if len(sub_a) > 0 else np.nan

            shock_compare_result.append({
                "shock_event": shock_name,
                "crop": crop,
                "variable": var,
                "mean_coef_before": mean_b,
                "mean_coef_after": mean_a,
                "delta": mean_a - mean_b
            })

df_shock_compare = pd.DataFrame(shock_compare_result)
df_shock_compare.to_excel("shock_all5_events_compare.xlsx", index=False)
print("✅ Sensitivity results saved to shock_all5_events_compare.xlsx")

for shock_name in shock_time_dict.keys():
    df_sub = df_shock_compare[df_shock_compare["shock_event"] == shock_name].copy()
    unique_crops = df_sub["crop"].unique()

    fig, axes = plt.subplots(nrows=len(unique_crops), ncols=1,
                             figsize=(15, 5 * len(unique_crops)), squeeze=False)
    fig.suptitle(f"Sensitivity Analysis: {shock_name} — Coefficients Before vs After Shock",
                 fontsize=14, y=0.98)

    for idx_crop, crop in enumerate(unique_crops):
        ax = axes[idx_crop, 0]
        df_c = df_sub[df_sub["crop"] == crop].copy()
        x_pos = np.arange(len(df_c))
        bar_width = 0.35

        ax.bar(x_pos - bar_width/2, df_c["mean_coef_before"], width=bar_width,
               label="Before shock", color="#636363")
        ax.bar(x_pos + bar_width/2, df_c["mean_coef_after"], width=bar_width,
               label="After shock", color="#c82423")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_c["variable"], rotation=45, ha="right", fontsize=9)
        ax.axhline(y=0, linestyle="--", color="black")
        ax.set_title(f"Crop: {crop}", fontsize=11)
        ax.set_ylabel("Average rolling‑window coefficient", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(f"sensitivity_{shock_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure:sensitivity_{shock_name}.png")

print("\n===== All event sensitivity analyses completed. =====")
