# 🛒 Retail Sales Big Data Dashboard 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PySpark](https://img.shields.io/badge/PySpark-Data_Processing-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![Machine Learning](https://img.shields.io/badge/MLlib-K--Means_Clustering-brightgreen)

## 📖 Tổng quan dự án (Project Overview)
Dự án này là một **Data Pipeline & Interactive Dashboard** mô phỏng quy trình xử lý dữ liệu lớn (Big Data) trong lĩnh vực bán lẻ (Retail). 
Mục tiêu của dự án là xây dựng một hệ thống hoàn chỉnh từ khâu trích xuất, làm sạch dữ liệu (ETL), trực quan hóa (Data Visualization) cho đến ứng dụng Trí tuệ nhân tạo (Machine Learning) để phân cụm khách hàng, giúp doanh nghiệp đưa ra các quyết định Marketing hiệu quả.

👉 **[Link xem Demo trực tuyến]** *(Nếu bạn có deploy lên Streamlit Cloud, hãy chèn link vào đây, nếu không hãy xóa dòng này)*

---

## 🛠️ Công nghệ sử dụng (Tech Stack)
* **Backend & Data Processing (ETL):** `Apache Spark (PySpark)`
* **Machine Learning:** `PySpark MLlib`
* **Frontend UI & Dashboard:** `Streamlit`
* **Data Manipulation:** `Pandas`
* **Data Visualization:** `Plotly Express`

---

## ✨ Các tính năng nổi bật (Key Features)

### 1. Kiến trúc xử lý dữ liệu chuẩn (Robust Data Pipeline)
* **Defensive Programming:** Sử dụng PySpark để định nghĩa Schema, tự động loại bỏ rác (White spaces, Null/NaN) và ép kiểu dữ liệu an toàn.
* **Tối ưu hiệu suất:** Sử dụng `@st.cache_data` và `@st.cache_resource` để lưu trữ Spark Session và Data vào bộ nhớ đệm, giúp giao diện web phản hồi mượt mà không độ trễ.

### 2. Giao diện Cuộn tương tác (One-Page Scroll Dashboard)
Thay vì chuyển trang thủ công, Dashboard được thiết kế chuẩn UX/UI liền mạch với thanh lọc dữ liệu (`Multiselect`) thông minh bên trái (Sidebar). Mọi thay đổi ở bộ lọc sẽ lập tức cập nhật toàn bộ biểu đồ.

### 3. Phân tích kinh doanh đa chiều (Business Analytics)
* **KPIs Tracking:** Theo dõi trực tiếp Tổng doanh thu, Số giao dịch, Lượng khách hàng và Giá trị trung bình/đơn (AOV).
* **Data Storytelling:** Trực quan hóa biến động doanh thu theo tháng, tỷ trọng danh mục sản phẩm và hành vi chi tiêu theo nhóm tuổi/giới tính thông qua các biểu đồ tương tác của `Plotly`.

### 4. Phân cụm khách hàng bằng AI (Customer Segmentation)
Ứng dụng thuật toán **K-Means Clustering (PySpark MLlib)** để tự động chia tập khách hàng thành 3 nhóm dựa trên Tổng chi tiêu và Tần suất mua hàng:
* 💎 **Nhóm VIP:** Số lượng mua ít nhưng giá trị rất cao -> Chiến lược chăm sóc đặc quyền.
* 🛍️ **Nhóm Trung cấp:** Mua lặt vặt đều đặn -> Chiến lược Cross-sell.
* 🏷️ **Nhóm Phổ thông:** Khách vãng lai, chi tiêu thấp -> Chiến lược gửi Email khuyến mãi kích cầu.

---

## 🚀 Hướng dẫn cài đặt và chạy dự án (Installation & Usage)

**Bước 1: Clone kho lưu trữ về máy**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
