
# HYBRID TOURISM RECOMMENDATION SYSTEM - PAPUA BARAT DAYA

## Model Information
- **Model Name**: Hybrid Tourism Recommender System
- **Version**: 1.0.0
- **Created Date**: 2026-01-02 06:15:05
- **Model Type**: Hybrid (70% Content-Based Filtering + 30% XGBoost)

## Model Components
1. **EnhancedPreprocessor** (`preprocessor.pkl`) - Preprocessing data wisata
2. **TF-IDF Vectorizer** (`tfidf_vectorizer.pkl`) - Text feature extraction
3. **MultiLabelBinarizer** (`mlb_kategori.pkl`) - Encoding kategori
4. **LabelEncoders** (`le_harga.pkl`, `le_lokasi.pkl`) - Encoding harga dan lokasi
5. **XGBoost Model** (`xgb_model.json`, `xgb_model.pkl`) - Popularity prediction
6. **Hybrid Recommender** (`hybrid_recommender.pkl`) - Full recommender system

## Data Statistics
- Total Wisata: 51
- Total Kategori: 45
- Average Rating: 4.41
- Average Minimum Price: Rp307,059

## Model Performance
- System Accuracy: 84.0%
- XGBoost Training R²: 0.9289
- XGBoost Testing R²: -0.3653

## Usage
1. Load the model using `joblib` or `pickle`
2. Use the `HybridTourismRecommender` for recommendations
3. Query examples available in `example_queries.json`

## Query Examples
- "wisata di kota sorong"
- "pantai dengan penginapan"
- "wisata murah di sorong selatan"
- "dimana pantai tanjung kasuari"

## File Structure
