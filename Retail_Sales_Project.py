import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, when, sum as spark_sum, to_date
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import pandas as pd
import plotly.express as px

# ==========================================
# 1. CẤU HÌNH TRANG
# ==========================================
st.set_page_config(page_title="Retail Big Data Portfolio", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    .main-header {font-size: 2rem; color: #1E3A8A; font-weight: bold; border-bottom: 2px solid #3B82F6; padding-bottom: 5px; margin-top: 40px;}
    .sub-header {font-size: 1.2rem; color: #4B5563; font-weight: bold; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU BACKEND (PYSPARK)
# ==========================================
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("Retail_Streamlit_Pro") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()

@st.cache_data
def load_and_transform_data():
    spark = get_spark_session()
    
    df = spark.read.csv(
        "D:\\Downloads\\Retail_Sales_Project\\Dataset\\retail_sales_dataset.csv", 
        header=True, 
        inferSchema=True,
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True
    )
    
    df_clean = df.withColumn("Date_Clean", to_date(col("Date"))) \
                 .dropna(subset=["Date_Clean", "Age", "Total Amount"])
    
    df_transformed = df_clean \
        .withColumn("Month", month(col("Date_Clean"))) \
        .withColumn("Age_Group", 
            when(col("Age") < 25, "1. 18-24")
            .when((col("Age") >= 25) & (col("Age") <= 35), "2. 25-35")
            .when((col("Age") > 35) & (col("Age") <= 50), "3. 36-50")
            .otherwise("4. 50+"))
            
    pdf = df_transformed.toPandas()
    pdf = pdf.dropna(subset=["Month", "Age", "Total Amount"])
    return pdf

@st.cache_data
def run_pyspark_kmeans(df_pd_input):
    spark = get_spark_session()
    df = spark.createDataFrame(df_pd_input)
    
    customer_df = df.groupBy("Customer ID").agg(
        spark_sum("Total Amount").alias("Total_Spend"),
        spark_sum("Quantity").alias("Total_Items")
    )
    
    assembler = VectorAssembler(inputCols=["Total_Spend", "Total_Items"], outputCol="features")
    feature_df = assembler.transform(customer_df)
    
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    scaled_df = scaler.fit(feature_df).transform(feature_df)
    
    kmeans = KMeans(featuresCol="scaledFeatures", k=3, seed=42)
    model = kmeans.fit(scaled_df)
    predictions = model.transform(scaled_df)
    
    pred_pd = predictions.select("Customer ID", "Total_Spend", "Total_Items", "prediction").toPandas()
    pred_pd['Customer_Type'] = pred_pd['prediction'].map({
        0: 'Nhóm phổ thông (Chi ít)', 
        1: 'Nhóm VIP (Chi cực nhiều)', 
        2: 'Nhóm Trung cấp (Mua đều đặn)'
    })
    return pred_pd

# ==========================================
# 3. SIDEBAR - BỘ LỌC MULTISELECT (HỘP CHỌN)
# ==========================================
df_pd = load_and_transform_data()

if df_pd.empty:
    st.error("🚨 Dữ liệu đầu vào bị lỗi. Hãy kiểm tra lại file CSV.")
    st.stop()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=100)
    st.markdown("### 🎯 BỘ LỌC DỮ LIỆU")
    
    # 1. Chọn Tháng (Dùng hộp chọn thay vì kéo)
    all_months = sorted(df_pd["Month"].unique().tolist())
    selected_months = st.multiselect("📅 Chọn Tháng:", options=all_months, default=all_months)
    if not selected_months: selected_months = all_months
    
    # 2. Chọn Danh mục
    all_categories = sorted(df_pd["Product Category"].unique().tolist())
    selected_category = st.multiselect("🏷️ Danh mục sản phẩm:", options=all_categories, default=all_categories)
    if not selected_category: selected_category = all_categories
        
    # 3. Chọn Nhóm tuổi (Dùng Age_Group trực quan hơn số tuổi gốc)
    all_age_groups = sorted(df_pd["Age_Group"].unique().tolist())
    selected_age = st.multiselect("👥 Nhóm tuổi:", options=all_age_groups, default=all_age_groups)
    if not selected_age: selected_age = all_age_groups
    
    # 4. Chọn Giới tính
    selected_gender = st.multiselect("🚻 Giới tính:", options=["Male", "Female"], default=["Male", "Female"])
    if not selected_gender: selected_gender = ["Male", "Female"]

# ==========================================
# ÁP DỤNG LOGIC LỌC TỔNG HỢP
# ==========================================
mask = (
    (df_pd["Month"].isin(selected_months)) &
    (df_pd["Product Category"].isin(selected_category)) &
    (df_pd["Age_Group"].isin(selected_age)) &
    (df_pd["Gender"].isin(selected_gender))
)
df_filtered = df_pd[mask]

# ==========================================
# THÂN TRANG - ONE-PAGE SCROLL DASHBOARD
# ==========================================
st.title("🛒 Báo cáo Dữ liệu Bán lẻ - Big Data Portfolio")
st.markdown("*Dự án ứng dụng **PySpark** cho xử lý dữ liệu lớn (ETL, Machine Learning) và **Streamlit** để thiết kế Dashboard tương tác.*")

if df_filtered.empty:
    st.warning("⚠️ Không có dữ liệu nào khớp với điều kiện lọc. Vui lòng chọn lại bộ lọc bên trái!")
    st.stop()

# ----------------------------------------
# PHẦN 1: TỔNG QUAN KINH DOANH
# ----------------------------------------
st.markdown('<p class="main-header">1. TỔNG QUAN KINH DOANH</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Tổng Doanh Thu", f"${df_filtered['Total Amount'].sum():,.0f}")
col2.metric("📦 Số Đơn Hàng", f"{df_filtered['Transaction ID'].nunique():,.0f}")
col3.metric("👥 Số Khách Hàng", f"{df_filtered['Customer ID'].nunique():,.0f}")
col4.metric("🛒 Giá Trị TB/Đơn", f"${df_filtered['Total Amount'].mean():,.2f}")

c1, c2 = st.columns(2)
with c1:
    fig_gender = px.pie(df_filtered, names='Gender', hole=0.5, title="Phân bổ Giới tính", color_discrete_sequence=['#3B82F6', '#EC4899'])
    st.plotly_chart(fig_gender, use_container_width=True)
with c2:
    age_counts = df_filtered['Age_Group'].value_counts().reset_index()
    fig_age = px.bar(age_counts, x='Age_Group', y='count', text_auto=True, title="Tỷ trọng theo Nhóm tuổi", color='Age_Group')
    st.plotly_chart(fig_age, use_container_width=True)

with st.expander("🔍 Click để xem dữ liệu thô (Raw Data) đã qua xử lý"):
    st.dataframe(df_filtered.head(100), use_container_width=True)

# ----------------------------------------
# PHẦN 2: PHÂN TÍCH CHUYÊN SÂU
# ----------------------------------------
st.markdown('<p class="main-header">2. PHÂN TÍCH DOANH THU & XU HƯỚNG</p>', unsafe_allow_html=True)

monthly_trend = df_filtered.groupby("Month")["Total Amount"].sum().reset_index().sort_values("Month")
fig_trend = px.area(monthly_trend, x="Month", y="Total Amount", markers=True, 
                    title="Biến động Doanh thu theo Tháng", color_discrete_sequence=['#10B981'])
# Ép Plotly hiển thị các tháng rời rạc (Ví dụ: Chọn 1, 3, 5 thì không vẽ tháng 2, 4)
fig_trend.update_xaxes(type='category') 
st.plotly_chart(fig_trend, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    cat_rev = df_filtered.groupby("Product Category")["Total Amount"].sum().reset_index()
    fig_cat = px.bar(cat_rev, x="Total Amount", y="Product Category", orientation='h', 
                     title="Doanh thu theo Danh mục", color="Product Category", text_auto='.2s')
    st.plotly_chart(fig_cat, use_container_width=True)

with c4:
    fig_scatter = px.scatter(df_filtered, x="Age", y="Total Amount", color="Gender", size="Quantity", 
                             title="Tương quan: Độ tuổi - Chi tiêu - Số lượng hàng")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------------------
# PHẦN 3: AI - MACHINE LEARNING
# ----------------------------------------
st.markdown('<p class="main-header">3. AI - PHÂN CỤM KHÁCH HÀNG (PYSPARK MLlib)</p>', unsafe_allow_html=True)
st.info("💡 **Góc nhìn Kỹ sư Dữ liệu:** Thuật toán K-Means tự động phân nhóm tập khách hàng đang được lọc bên trái để đưa ra chiến lược Marketing phù hợp.")

# K-Means cần ít nhất 3 khách hàng để chia làm 3 cụm, ta cần bẫy lỗi này
if df_filtered["Customer ID"].nunique() < 3:
    st.warning("⚠️ Dữ liệu hiện tại quá ít (Dưới 3 khách hàng) để có thể chạy mô hình K-Means. Vui lòng chọn thêm dữ liệu ở bộ lọc.")
else:
    with st.spinner("Đang chạy mô hình AI trên nền tảng PySpark..."):
        kmeans_results = run_pyspark_kmeans(df_filtered)
    
    c5, c6 = st.columns([2, 1])
    with c5:
        fig_cluster = px.scatter(kmeans_results, x="Total_Spend", y="Total_Items", color="Customer_Type", 
                                 size="Total_Spend", title="Bản đồ phân cụm Khách hàng",
                                 labels={"Total_Spend": "Tổng chi tiêu ($)", "Total_Items": "Tổng SP đã mua"})
        st.plotly_chart(fig_cluster, use_container_width=True)
        
    with c6:
        st.markdown("<br>", unsafe_allow_html=True) # Khoảng trắng cho cân đối biểu đồ
        st.success("**Nhóm VIP:** Số lượng mua ít nhưng giá trị rất cao. Chăm sóc đặc quyền, tặng voucher.")
        st.info("**Nhóm Trung cấp:** Mua lặt vặt đều đặn. Cần chiến lược bán chéo (Cross-sell).")
        st.warning("**Nhóm Phổ thông:** Khách vãng lai, chi tiêu thấp. Cần Sale kích cầu.")