# import os
# import pandas as pd
# for dirname, _, filenames in os.walk('D:\data\causalgraph2025\data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# df = pd.read_csv('/kaggle/input/apple-quality/apple_quality.csv')
# df.head()
# print(df)




import pandas as pd
from typing import Optional, Sequence, Dict
from pandas.api.types import is_numeric_dtype
from typing import List, Sequence

def build_agri_causal_dataset(
    crop_csv: str,
    soil_csv: str,
    weather_csv: str,
    output_csv: Optional[str] = None,
    # ---- 列名 / 逻辑相关超参 ----
    crop_key_cols: Sequence[str] = ("state", "year", "crop", "season"),
    soil_key_state_col: str = "state",
    soil_key_year_col: str = "year",
    soil_value_cols: Sequence[str] = ("n", "p", "k", "ph"),
    weather_key_state_col: str = "state",
    weather_key_year_col: str = "year",
    weather_date_col: str = "date",  # 如果没有 year，但有日期，就用它提取 year
    weather_value_cols: Dict[str, str] = None,
    # 默认天气聚合方式：温度/湿度取平均，降雨取总和
    drop_rows_missing_weather: bool = True,
    auto_compute_yield: bool = True,
    yield_col: str = "yield",
    production_col: str = "production",
    area_col: str = "area",
    verbose: bool = True,
):
    """
    构建统一的农业因果发现数据集：
    - 读取 crop / soil / weather 三个 csv
    - 清洗列名、state、year
    - 土壤按 state 或 (state, year) 聚合
    - 天气按 (state, year) 聚合（daily/monthly/yearly 都兼容）
    - 最终返回合并后的 DataFrame，并可选保存为 csv

    参数说明：
    ----------
    crop_csv, soil_csv, weather_csv : 三个原始 csv 路径
    output_csv : 可选，若给定则把合并后的数据保存到此路径
    crop_key_cols : 用来检查 crop 表是否有重复键的列
    soil_value_cols : 土壤中要保留/聚合的数值列
    weather_value_cols : dict 映射 {列名: 聚合方式}，如
        {"avg_temp_c": "mean", "total_rainfall_mm": "sum", "avg_humidity_percent": "mean"}
        若为 None，则自动在表中查找同名列并使用该默认策略
    drop_rows_missing_weather : 是否丢弃天气缺失的行
    auto_compute_yield : 若没有 yield 列或全是 NaN，是否用 production/area 计算
    verbose : 是否打印中间信息

    返回：
    ----
    df_final : pandas.DataFrame
    """

    # ---------- 小工具：加载并清洗列名 / state ----------
    def load_and_clean(path, key_cols=None):
        df = pd.read_csv(path)
        # 统一列名：去空格 -> 小写 -> 空格/横线改下划线
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(r"\s+", "_", regex=True)
                      .str.replace("-", "_")
        )
        # 关键列：strip + 对 state 做小写
        if key_cols:
            for c in key_cols:
                if c in df.columns:
                    df[c] = df[c].astype(str).str.strip()
                    if c == "state":
                        df[c] = df[c].str.lower()
        return df

    # ---------- 1. 读取三张表 ----------
    crop_df = load_and_clean(crop_csv, key_cols=crop_key_cols)
    soil_df = load_and_clean(soil_csv, key_cols=[soil_key_state_col, soil_key_year_col])
    weather_df = load_and_clean(weather_csv, key_cols=[weather_key_state_col, weather_key_year_col, weather_date_col])

    # ---------- 2. year 类型统一 ----------
    for df in (crop_df, soil_df, weather_df):
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # ---------- 3. 检查必需列 ----------
    def ensure_cols(df, cols, name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} 缺少列: {missing}")

    # crop 端：这些是最基本的
    ensure_cols(crop_df,
                ["crop", "year", "season", "state", area_col, production_col],
                "crop_df")

    # soil：N/P/K/pH
    ensure_cols(soil_df, [soil_key_state_col] + list(soil_value_cols), "soil_df")

    # weather 至少要有 state + (year 或 date)
    if "year" not in weather_df.columns and weather_date_col not in weather_df.columns:
        raise ValueError("weather_df 既没有 'year' 列也没有 date 列，无法按年份聚合")

    # ---------- 4. 处理土壤 soil_df ----------
    # 判断是否存在 year 列
    if soil_key_year_col in soil_df.columns and soil_key_year_col in soil_df:
        # 看 (state, year) 是否有重复
        has_dup_state_year = soil_df.duplicated(subset=[soil_key_state_col, soil_key_year_col]).any()
        if has_dup_state_year:
            # 多行 -> 按 state+year 聚合
            soil_agg = (
                soil_df
                .groupby([soil_key_state_col, soil_key_year_col], as_index=False)[list(soil_value_cols)]
                .mean()
            )
            soil_merge_keys = [soil_key_state_col, soil_key_year_col]
            if verbose:
                print("Soil: 按 (state, year) 聚合")
        else:
            # 没有重复 -> 按 state 聚合成多年平均
            soil_agg = (
                soil_df
                .groupby(soil_key_state_col, as_index=False)[list(soil_value_cols)]
                .mean()
            )
            soil_merge_keys = [soil_key_state_col]
            if verbose:
                print("Soil: 按 state 聚合为多年平均")
    else:
        # 没 year -> 默认按 state 聚合
        soil_agg = (
            soil_df
            .groupby(soil_key_state_col, as_index=False)[list(soil_value_cols)]
            .mean()
        )
        soil_merge_keys = [soil_key_state_col]
        if verbose:
            print("Soil: 仅按 state 聚合")

    if verbose:
        print("soil_agg head:")
        print(soil_agg.head())

    # ---------- 5. 处理天气 weather_df ----------
    # 如果没有 year 但有 date，用 date 提取 year
    if "year" not in weather_df.columns and weather_date_col in weather_df.columns:
        weather_df[weather_date_col] = pd.to_datetime(weather_df[weather_date_col])
        weather_df["year"] = weather_df[weather_date_col].dt.year

    ensure_cols(weather_df, [weather_key_state_col, weather_key_year_col], "weather_df (for yearly agg)")

    # 默认天气聚合策略
    if weather_value_cols is None:
        # 尝试使用你之前提到的列名
        default_cols = {
            "avg_temp_c": "mean",
            "total_rainfall_mm": "sum",
            "avg_humidity_percent": "mean",
        }
        # 只保留表中实际存在的列
        weather_value_cols = {c: agg for c, agg in default_cols.items() if c in weather_df.columns}

    if not weather_value_cols:
        raise ValueError("weather_value_cols 为空，天气表中找不到任何可聚合的数值列")

    weather_yearly = (
        weather_df
        .groupby([weather_key_state_col, weather_key_year_col], as_index=False)
        .agg(weather_value_cols)
    )

    if verbose:
        print("weather_yearly head:")
        print(weather_yearly.head())

    # ---------- 6. 检查 crop_df 是否有重复键 ----------
    if set(crop_key_cols) <= set(crop_df.columns):
        if crop_df.duplicated(subset=list(crop_key_cols)).any():
            raise ValueError(
                f"crop_df 中 {crop_key_cols} 出现多行，"
                f"需要先自行决定如何聚合（比如按地区再求和或取平均）。"
            )

    # ---------- 7. 合并 crop + soil ----------
    df = crop_df.merge(
        soil_agg,
        how="left",
        left_on=soil_merge_keys,
        right_on=soil_merge_keys,
        validate="m:1",  # 每个 crop 记录对应最多一条 soil 记录
    )

    # ---------- 8. 合并 weather ----------
    df = df.merge(
        weather_yearly,
        how="left",
        left_on=[weather_key_state_col, weather_key_year_col],
        right_on=[weather_key_state_col, weather_key_year_col],
        validate="m:1",  # 每个 crop 记录对应最多一条 weather 记录
    )

    if verbose:
        print("Merged shape:", df.shape)

    # ---------- 9. 自动计算 yield ----------
    if auto_compute_yield:
        if (yield_col not in df.columns) or df[yield_col].isna().all():
            if production_col in df.columns and area_col in df.columns:
                df[yield_col] = df[production_col] / df[area_col]
                if verbose:
                    print(f"Auto computed `{yield_col}` = {production_col} / {area_col}")
            else:
                if verbose:
                    print("无法自动计算 yield：缺少 production 或 area 列")

    # ---------- 10. 视情况丢弃天气缺失行 ----------
    if drop_rows_missing_weather and weather_value_cols:
        must_have_cols = list(weather_value_cols.keys())
        before = len(df)
        df = df.dropna(subset=must_have_cols, how="any")
        if verbose:
            print(f"Dropped rows with missing weather: {before} -> {len(df)}")

    if verbose:
        print("Final head:")
        print(df.head())

    # ---------- 11. 输出 csv ----------
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"Saved merged dataset to: {output_csv}")

    return df



DEFAULT_DISCRETE_COLS = ("state", "crop", "season", "year")


def get_continuous_columns(
    df: pd.DataFrame,
    exclude_cols: Sequence[str] = None,
    min_unique: int = 5,
    verbose: bool = True,
) -> List[str]:
    """
    筛选“连续型特征列”：
    1. 数值类型 (int/float)
    2. 唯一值个数 >= min_unique
    3. 排除 exclude_cols（默认把 state/crop/season/year 当成离散）

    返回：连续列名列表
    """
    # 如果用户没传 exclude_cols，就默认把这些当离散：
    # state, crop, season, year
    if exclude_cols is None:
        exclude_cols = DEFAULT_DISCRETE_COLS

    exclude_set = set(exclude_cols)
    continuous_cols = []

    for col in df.columns:
        if col in exclude_set:
            continue
        s = df[col]

        # 必须是数值型
        if not is_numeric_dtype(s):
            continue

        # 唯一值太少则视作离散编码，不当作连续
        nunique = s.nunique(dropna=True)
        if nunique < min_unique:
            continue

        continuous_cols.append(col)

    if verbose:
        print(f"检测到连续型特征列 {len(continuous_cols)} 个：")
        print(continuous_cols)

    return continuous_cols
