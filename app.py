import streamlit as st
import pickle
import pandas as pd

# ==========================
# Konfigurasi Halaman
# ==========================

st.set_page_config(
    page_title="Amazon Product Recommendation",
    page_icon="🛒",
    layout="wide"
)

# ==========================
# Load Data
# ==========================

with open("recommendation_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

df = artifacts["df_products"]
similarity = artifacts["content_similarity"]      # gunakan ndarray
tfidf = artifacts["tfidf_vectorizer"]

# ==========================
# Fungsi Rekomendasi
# ==========================
def recommend(product_name, top_n=20):

    # Cari index produk
    idx = df[df["product_name"] == product_name].index[0]

    # Hitung similarity semua produk
    sim_scores = list(enumerate(similarity[idx]))

    # Urutkan berdasarkan similarity
    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Buang produk yang dipilih sendiri
    sim_scores = sim_scores[1:101]      # Ambil 100 produk paling mirip

    hasil = []

    for i, score in sim_scores:

        row = df.iloc[i].copy()

        # Bersihkan rating
        rating = str(row["rating"])
        try:
            rating = float(rating)
        except:
            rating = pd.to_numeric(
                rating.replace("|", "").split()[0],
                errors="coerce"
            )

        row["rating"] = rating

        # Bersihkan harga
        harga = (
            str(row["discounted_price"])
            .replace("₹", "")
            .replace(",", "")
        )

        row["discounted_price"] = pd.to_numeric(
            harga,
            errors="coerce"
        )

        row["Similarity"] = float(score)

        hasil.append(row)

    hasil = pd.DataFrame(hasil)

    # Normalisasi similarity
    hasil["Similarity_norm"] = (
        hasil["Similarity"] - hasil["Similarity"].min()
    ) / (
        hasil["Similarity"].max() - hasil["Similarity"].min()
    )

    # Normalisasi rating
    hasil["Rating_norm"] = (
        hasil["rating"] - hasil["rating"].min()
    ) / (
        hasil["rating"].max() - hasil["rating"].min()
    )

    # Skor akhir
    hasil["Final Score"] = (
        0.7 * hasil["Similarity_norm"] +
        0.3 * hasil["Rating_norm"]
    )

    # Urutkan berdasarkan skor akhir
    hasil = hasil.sort_values(
    by=["rating", "Similarity"],
    ascending=[False, False]
    )

    # Ambil top N
    hasil = hasil.head(top_n)

    # Format angka
    hasil["Similarity"] = hasil["Similarity"].round(3)
    hasil["rating"] = hasil["rating"].round(1)

    return hasil


# ==========================
# Header
# ==========================

st.title("🛒 Sistem Rekomendasi Produk Amazon")

st.write(
    "Cari nama produk kemudian pilih salah satu produk untuk memperoleh rekomendasi."
)

# ==========================
# Search
# ==========================

keyword = st.text_input(
    "🔍 Cari Produk",
    placeholder="Contoh : iphone, samsung, keyboard, mouse..."
)

if keyword == "":

    daftar_produk = sorted(df["product_name"].unique())

else:

    daftar_produk = sorted(
        df[
            df["product_name"]
            .str.contains(keyword,
                          case=False,
                          na=False)
        ]["product_name"].unique()
    )

if len(daftar_produk) == 0:

    st.warning("Produk tidak ditemukan.")

    st.stop()

# ==========================
# Pilih Produk
# ==========================

produk = st.selectbox(
    "Pilih Produk",
    daftar_produk
)

# ==========================
# Info Produk
# ==========================

produk_info = df[df["product_name"]==produk].iloc[0]

st.subheader("Produk yang Dipilih")

col1,col2,col3=st.columns(3)

with col1:

    if "category_main" in df.columns:
        st.metric("Kategori",produk_info["category_main"])

with col2:

    if "rating" in df.columns:
        st.metric("Rating ⭐",produk_info["rating"])

with col3:

    if "discounted_price" in df.columns:
        st.metric("Harga",produk_info["discounted_price"])

# ==========================
# Tombol
# ==========================

if st.button("🎯 Cari Rekomendasi"):

    hasil = recommend(produk)

    st.success(f"Top {len(hasil)} Produk yang Direkomendasikan")

    st.dataframe(
    hasil[
        [
            "product_name",
            "category_main",
            "rating",
            "discounted_price",
            "Similarity",
            "Final Score"
        ]
    ].reset_index(drop=True),
    use_container_width=True
)