import pandas as pd
import joblib
import json
import os
import re
import pickle
import numpy as np
import difflib
from datetime import datetime
from collections import Counter
import zipfile
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import nltk
from nltk.corpus import stopwords
from rapidfuzz import fuzz, process
import sys  # Tambahkan ini untuk hack module

warnings.filterwarnings('ignore')

# Download required NLTK data if needed
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# === Slugify Function ===
def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)  # Hapus simbol
    text = re.sub(r'\s+', '-', text)     # Spasi → -
    return text

# TAMBAHKAN DEFINISI KELAS FixedPreprocessor DARI COLAB
class FixedPreprocessor:
    """Preprocessor dengan perbaikan untuk handling data yang lebih baik"""

    def __init__(self):
        self.location_synonyms = {
            'sorong': ['kota sorong', 'sorong kota', 'sorong'],
            'maladumes': ['maladumes', 'maladum mes'],
            'klawasi': ['klawasi'],
            'sorong_selatan': ['sorsel', 'sorong selatan'],
            'teminabuan': ['teminabuan'],
            'maybrat': ['maybrat', 'ayamaru'],
            'raja_ampat': ['raja ampat', 'rajaampat'],
            'misool': ['misool'],
            'waigeo': ['waigeo'],
            'makbon': ['makbon'],
            'klamono': ['klamono'],
            'bal_bulol': ['bal bulol'],
            'love_lagoon': ['love lagoon', 'lovelagoon'],
            'telaga_bintang': ['telaga bintang'],
            'kali_biru': ['kali biru', 'blue river'],
            'wayag': ['wayag'],
            'pasir_timbul': ['pasir timbul']
        }

        self.category_mapping = {
            'pantai': ['pantai', 'beach', 'tanjung', 'pesisir'],
            'sungai': ['sungai', 'kali', 'river'],
            'danau': ['danau', 'telaga', 'lake'],
            'air_terjun': ['air terjun', 'waterfall', 'curug'],
            'pulau': ['pulau', 'island'],
            'taman': ['taman', 'park'],
            'bukit': ['bukit', 'hill'],
            'gunung': ['gunung', 'mountain'],
            'goa': ['goa', 'gua', 'cave'],
            'kolam': ['kolam', 'pool'],
            'laguna': ['laguna', 'lagoon'],
            'spot_foto': ['spot foto', 'photo spot'],
            'spot_sunset': ['spot sunset', 'sunset spot'],
            'wisata_alam': ['wisata alam', 'nature'],
            'snorkeling': ['snorkeling', 'diving'],
            'geologi': ['geologi', 'geological'],
            'taman_rekreasi': ['taman rekreasi'],
            'resort': ['resort']
        }

        self.facility_keywords = {
            'penginapan': ['penginapan', 'hotel', 'villa', 'resort', 'cottage', 'homestay'],
            'kolam_renang': ['kolam renang', 'swimming pool', 'pool'],
            'restoran': ['restoran', 'restaurant', 'warung', 'kafe'],
            'parkir': ['parkir', 'parking'],
            'toilet': ['toilet', 'wc', 'kamar mandi'],
            'wifi': ['wifi', 'internet']
        }

    def normalize_location(self, location_data: dict) -> dict:
        """Normalisasi lokasi"""
        if not location_data:
            return {'kecamatan': '', 'kabupaten_kota': '', 'provinsi': ''}

        kecamatan = location_data.get('kecamatan', '') or ''
        kabupaten = location_data.get('kabupaten', '') or location_data.get('kabupaten_kota', '') or ''
        provinsi = location_data.get('provinsi', '') or 'Papua Barat Daya'

        # Standardisasi
        kecamatan = kecamatan.strip().title()
        kabupaten = kabupaten.strip().title()
        provinsi = provinsi.strip().title()

        return {
            'kecamatan': kecamatan,
            'kabupaten_kota': kabupaten,
            'provinsi': provinsi
        }

    def extract_categories(self, nama: str, deskripsi: str, existing_categories: list) -> list:
        """Ekstrak kategori dari data"""
        categories = set()

        # Tambahkan kategori existing
        if existing_categories:
            if isinstance(existing_categories, str):
                existing_categories = [existing_categories]
            for cat in existing_categories:
                cat_lower = str(cat).lower().strip()
                categories.add(cat_lower)

        # Deteksi dari nama
        nama_lower = nama.lower()
        for cat_name, synonyms in self.category_mapping.items():
            for synonym in synonyms:
                if synonym in nama_lower:
                    categories.add(cat_name)

        return list(categories)

    def extract_facilities(self, fasilitas_data: any) -> dict:
        """Ekstrak fasilitas"""
        facilities = set()
        facility_flags = {
            'penginapan': 0,
            'kolam_renang': 0,
            'restoran': 0,
            'parkir': 0,
            'toilet': 0,
            'wifi': 0
        }

        if not fasilitas_data:
            return {'list': [], 'flags': facility_flags}

        # Konversi ke string
        if isinstance(fasilitas_data, list):
            fasilitas_str = ' '.join([str(f).lower() for f in fasilitas_data])
        else:
            fasilitas_str = str(fasilitas_data).lower()

        # Parsing fasilitas
        for fac_type, keywords in self.facility_keywords.items():
            for keyword in keywords:
                if keyword in fasilitas_str:
                    facilities.add(fac_type)
                    facility_flags[fac_type] = 1
                    break

        return {'list': list(facilities), 'flags': facility_flags}

    def process_dataframe(self, raw_data: dict) -> pd.DataFrame:
        """Proses data menjadi DataFrame"""
        records = []

        for key, item in raw_data.items():
            record = item.copy()

            # Extract fitur turunan
            turunan = record.pop('fitur_turunan', {})
            record.update(turunan)

            # Normalisasi lokasi
            lokasi = self.normalize_location(record.get('lokasi', {}))
            record['lokasi'] = lokasi

            # Ekstrak kategori
            categories = self.extract_categories(
                record.get('nama', ''),
                record.get('deskripsi', ''),
                record.get('kategori', [])
            )
            record['kategori'] = categories

            # Ekstrak fasilitas
            facilities = self.extract_facilities(record.get('fasilitas', ''))
            record['fasilitas_list'] = facilities['list']
            record.update(facilities['flags'])

            # Parse harga
            try:
                harga_min = float(record.get('harga_min', 0) or 0)
                harga_max = float(record.get('harga_max', 0) or 0)
            except:
                harga_min = 0
                harga_max = 0

            record['harga_min'] = harga_min
            record['harga_max'] = harga_max

            # Tentukan level harga
            if harga_min == 0:
                record['harga_level'] = 'gratis'
            elif harga_min <= 20000:
                record['harga_level'] = 'sangat_murah'
            elif harga_min <= 50000:
                record['harga_level'] = 'murah'
            elif harga_min <= 100000:
                record['harga_level'] = 'sedang'
            elif harga_min <= 300000:
                record['harga_level'] = 'mahal'
            else:
                record['harga_level'] = 'sangat_mahal'

            # Numeric features
            record['rating'] = float(record.get('rating', 0)) if record.get('rating') not in [None, ''] else 0
            record['jumlah_review'] = int(record.get('jumlah_review', 0)) if record.get('jumlah_review') not in [None, ''] else 0

            records.append(record)

        df = pd.DataFrame(records)

        # Tambahkan fitur tambahan
        df['pop_score'] = df['rating'] * (1 + df['jumlah_review'] / 100)
        df['lokasi_str'] = df['lokasi'].apply(lambda x: f"{x['kecamatan']} {x['kabupaten_kota']}".lower())

        return df

# TAMBAHKAN DEFINISI KELAS ImprovedQueryParser DARI COLAB
class ImprovedQueryParser:
    """Parser query yang lebih akurat dengan NLP"""

    def __init__(self, preprocessor: FixedPreprocessor):
        self.preprocessor = preprocessor
        self.stop_words = set(stopwords.words('indonesian'))

    def parse_query(self, query: str) -> dict:
        """Parse query menjadi structured intent"""
        query_lower = query.lower().strip()

        # Simple tokenization without NLTK
        tokens = query_lower.split()
        tokens = [word for word in tokens if word not in self.stop_words]

        result = {
            'original_query': query,
            'intent': 'unknown',
            'target_name': '',
            'locations': [],
            'categories': [],
            'price_filters': [],
            'facility_filters': [],
            'negation_filters': [],
            'processed_tokens': tokens
        }

        # Deteksi intent utama
        intent_keywords = {
            'location': ['dimana', 'mana', 'lokasi', 'letak'],
            'price': ['harga', 'biaya', 'berapa', 'rp', 'ribu', 'juta'],
            'facility': ['fasilitas', 'ada', 'punya', 'tersedia'],
            'recommendation': ['rekomendasi', 'sarankan', 'cari', 'tempat', 'wisata']
        }

        for intent, keywords in intent_keywords.items():
            if any(word in tokens for word in keywords):
                result['intent'] = intent
                break
        else:
            result['intent'] = 'search'

        # Ekstrak nama target untuk query lokasi
        if result['intent'] == 'location':
            cleaned = re.sub(r'(dimana|di mana|lokasi|letak|ada\s+dimana|\?)\s*', '', query_lower).strip()
            result['target_name'] = cleaned

        # Ekstrak lokasi dengan fuzzy matching
        for loc_name, synonyms in self.preprocessor.location_synonyms.items():
            all_terms = [loc_name] + synonyms
            for term in all_terms:
                term_lower = term.lower()
                # Check for exact match in query
                if term_lower in query_lower:
                    if loc_name not in result['locations']:
                        result['locations'].append(loc_name)
                    break
                # Check partial match
                elif any(term_lower in token or token in term_lower for token in tokens):
                    if loc_name not in result['locations']:
                        result['locations'].append(loc_name)
                    break

        # Ekstrak kategori
        for cat_name, synonyms in self.preprocessor.category_mapping.items():
            for synonym in synonyms:
                synonym_lower = synonym.lower()
                if synonym_lower in query_lower:
                    if cat_name not in result['categories']:
                        result['categories'].append(cat_name)
                    break
                elif any(fuzz.ratio(synonym_lower, token) > 80 for token in tokens):
                    if cat_name not in result['categories']:
                        result['categories'].append(cat_name)
                    break

        # Ekstrak filter harga
        if 'gratis' in query_lower or 'free' in query_lower:
            result['price_filters'].append({'type': 'free', 'min': 0, 'max': 0})

        if 'murah' in query_lower or 'hemat' in query_lower:
            result['price_filters'].append({'type': 'cheap', 'min': 0, 'max': 50000})

        if 'mahal' in query_lower or 'expensive' in query_lower:
            result['price_filters'].append({'type': 'expensive', 'min': 100000, 'max': float('inf')})

        # Ekstrak range harga
        price_patterns = [
            (r'dibawah\s*(\d+)\s*(?:ribu|rb|k)', 'max'),
            (r'kurang\s+dari\s+(\d+)\s*(?:ribu|rb|k)', 'max'),
            (r'diatas\s+(\d+)\s*(?:ribu|rb|k)', 'min'),
            (r'(\d+)\s*[-–]\s*(\d+)\s*(?:ribu|rb|k)', 'range')
        ]

        for pattern, price_type in price_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                for match in matches:
                    if price_type == 'max':
                        try:
                            value = int(match) * 1000
                            result['price_filters'].append({'type': 'max', 'max': value})
                        except:
                            pass
                    elif price_type == 'min':
                        try:
                            value = int(match) * 1000
                            result['price_filters'].append({'type': 'min', 'min': value})
                        except:
                            pass
                    elif price_type == 'range':
                        if isinstance(match, tuple) and len(match) == 2:
                            try:
                                min_val = int(match[0]) * 1000
                                max_val = int(match[1]) * 1000
                                result['price_filters'].append({'type': 'range', 'min': min_val, 'max': max_val})
                            except:
                                pass

        # Ekstrak filter fasilitas dengan negasi
        negation_words = ['tanpa', 'tidak', 'no', 'belum']

        for fac_name, keywords in self.preprocessor.facility_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Cek jika ada negasi sebelum keyword
                    words = query_lower.split()
                    keyword_index = -1
                    for i, word in enumerate(words):
                        if keyword in word:
                            keyword_index = i
                            break

                    if keyword_index != -1:
                        # Check for negation in surrounding words
                        has_negation = any(
                            neg_word in words[max(0, keyword_index-2):keyword_index+1]
                            for neg_word in negation_words
                        )

                        if has_negation:
                            if fac_name not in result['negation_filters']:
                                result['negation_filters'].append(fac_name)
                        else:
                            if fac_name not in result['facility_filters']:
                                result['facility_filters'].append(fac_name)

        return result

# Set module to match pickle from Colab
FixedPreprocessor.__module__ = 'recommender'
ImprovedQueryParser.__module__ = 'recommender'

# Ekstrak ZIP jika belum
MODEL_ZIP = 'models.zip'
MODEL_DIR = 'model_wisata_recommender_x2'

if os.path.exists(MODEL_ZIP) and not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zipf:
        zipf.extractall(MODEL_DIR)

# Ubah nama file dan path sesuai dengan yang disimpan di Colab
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.json')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
MLB_KATEGORI_PATH = os.path.join(MODEL_DIR, 'mlb_kategori.pkl')
LE_HARGA_PATH = os.path.join(MODEL_DIR, 'le_harga.pkl')
LE_LOKASI_PATH = os.path.join(MODEL_DIR, 'le_lokasi.pkl')
PROCESSED_DF_PATH = os.path.join(MODEL_DIR, 'processed_df.pkl')
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
QUERY_PARSER_PATH = os.path.join(MODEL_DIR, 'query_parser.pkl')
NUMERIC_FEATURES_PATH = os.path.join(MODEL_DIR, 'numeric_features.npy')
TFIDF_FEATURES_PATH = os.path.join(MODEL_DIR, 'tfidf_features.npy')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
TOP_KEYWORDS_PATH = os.path.join(MODEL_DIR, 'top_keywords.pkl')
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, 'embeddings.pkl')

try:
    # Muat model XGBoost
    model_xgb = xgb.XGBRegressor()
    model_xgb.load_model(XGB_MODEL_PATH)
    
    # Muat TF-IDF Vectorizer
    tfidf = joblib.load(TFIDF_VECTORIZER_PATH)
    
    # Muat MultiLabelBinarizer untuk kategori
    mlb = joblib.load(MLB_KATEGORI_PATH)
    
    # Muat LabelEncoder untuk harga
    le_harga = joblib.load(LE_HARGA_PATH)
    
    # Muat LabelEncoder untuk lokasi
    le_loc = joblib.load(LE_LOKASI_PATH)
    
    # Muat data wisata
    df = pd.read_pickle(PROCESSED_DF_PATH)
    
    # Muat konfigurasi
    with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    # Dapatkan available features dari model_info
    available_features = model_info.get('available_features', [])
    
    # Hack untuk module mismatch sebelum memuat preprocessor
    old_main = sys.modules['__main__']
    sys.modules['__main__'] = sys.modules[__name__]  # Set to current module ('recommender')
    
    # Muat preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Muat query parser
    parser = joblib.load(QUERY_PARSER_PATH)
    
    # Kembalikan sys.modules['__main__']
    sys.modules['__main__'] = old_main
    
    # Muat numeric features
    numeric_features = np.load(NUMERIC_FEATURES_PATH)
    
    # Muat TF-IDF features
    tfidf_matrix = np.load(TFIDF_FEATURES_PATH)
    
    # Muat scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Muat top keywords
    with open(TOP_KEYWORDS_PATH, 'rb') as f:
        top_keywords = pickle.load(f)
    
    # Muat embeddings
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings_array = pickle.load(f)
    
except FileNotFoundError as e:
    raise FileNotFoundError(f"File tidak ditemukan: {str(e)}")
except Exception as e:
    raise RuntimeError(f"Error saat memuat model: {str(e)}")

# === TAMBAHKAN KOLOM YANG DIBUTUHKAN ===
# Tambahkan kolom yang diperlukan dari Colab jika belum ada
if 'processed_deskripsi' not in df.columns:
    stop_words = set(stopwords.words('indonesian'))
    def preprocess_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    df['processed_deskripsi'] = df['deskripsi'].apply(preprocess_text)

# Tambahkan harga_avg jika belum ada
if 'harga_avg' not in df.columns:
    df['harga_avg'] = (df['harga_min'] + df['harga_max']) / 2

# Tambahkan lokasi_encoded jika belum ada
if 'lokasi_encoded' not in df.columns:
    df['lokasi_encoded'] = le_loc.transform(df['lokasi_str'])

# Tambahkan kolom embedding features jika perlu (gunakan embeddings_array dari load)
for i in range(embeddings_array.shape[1]):
    col_name = f'emb_{i}'
    if col_name not in df.columns:
        df[col_name] = embeddings_array[:, i]

# Tambahkan popularity_score dan target_score seperti di VSCode
df['popularity_score'] = np.log1p(df['jumlah_review'])
df['target_score'] = df['rating'] * (1 + df['popularity_score'] / 10)

# === DEFINISI fac_cols dan turunan_cols ===
fac_cols = ['penginapan', 'kolam_renang', 'restoran', 'parkir', 'toilet', 'wifi']

turunan_cols = [
    'punya_kolam_renang', 'target_keluarga', 'is_pantai', 'spot_foto',
    'buka_24jam', 'wisata_edukasi', 'spot_sunset', 'aktivitas_air',
    'aksesibilitas', 'eksotis_papua', 'air_jernih', 'wisata_bahari',
    'suasana_tenang', 'wisata_eksklusif', 'wisata_ekstrem', 'eco_tourism',
    'keunikan', 'spot_panorama'
]

# === CLASS IMPROVED RECOMMENDER (DARI COLAB, DISESUAIKAN UNTUK RETURN CARDS) ===
class ImprovedRecommender:
    """Sistem rekomendasi dengan perbaikan dari Colab, disesuaikan untuk return cards"""

    def __init__(self, df: pd.DataFrame, parser, numeric_features, tfidf_features, xgb_model=None, tfidf_vectorizer=None):
        self.df = df
        self.parser = parser
        self.numeric_features = numeric_features
        self.tfidf_features = tfidf_features
        self.xgb_model = xgb_model
        self.tfidf_vectorizer = tfidf_vectorizer

        # Bangun indeks untuk pencarian cepat
        self._build_indexes()

        # Inisialisasi history dan stats (pertahankan dari VSCode asli)
        self.query_history = []
        self.recommendation_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0
        }

    def _build_indexes(self):
        """Bangun indeks untuk pencarian cepat"""
        self.name_index = {}
        self.category_index = {}
        self.location_index = {}
        self.facility_index = {}

        for idx, row in self.df.iterrows():
            # Index nama
            name = row['nama'].lower()
            self.name_index[name] = idx

            # Index kategori
            for category in row['kategori']:
                if category not in self.category_index:
                    self.category_index[category] = []
                self.category_index[category].append(idx)

            # Index lokasi
            lokasi = row['lokasi']
            kecamatan = lokasi['kecamatan'].lower()
            kabupaten = lokasi['kabupaten_kota'].lower()

            if kecamatan:
                if kecamatan not in self.location_index:
                    self.location_index[kecamatan] = []
                self.location_index[kecamatan].append(idx)

            if kabupaten:
                if kabupaten not in self.location_index:
                    self.location_index[kabupaten] = []
                self.location_index[kabupaten].append(idx)

            # Index fasilitas
            for facility in ['penginapan', 'kolam_renang', 'restoran', 'parkir', 'toilet', 'wifi']:
                if row.get(facility, 0) == 1:
                    if facility not in self.facility_index:
                        self.facility_index[facility] = []
                    self.facility_index[facility].append(idx)

    def _find_by_name(self, name: str) -> list:
        """Cari wisata berdasarkan nama dengan fuzzy matching"""
        name_lower = name.lower().strip()

        # 1. Exact match
        if name_lower in self.name_index:
            return [self.name_index[name_lower]]

        # 2. Partial match
        partial_matches = []
        for stored_name, idx in self.name_index.items():
            if name_lower in stored_name or stored_name in name_lower:
                partial_matches.append(idx)

        if partial_matches:
            return partial_matches

        # 3. Fuzzy match
        all_names = list(self.name_index.keys())
        results = process.extract(name_lower, all_names, limit=3)

        matches = []
        for result_name, score, _ in results:
            if score >= 70:  # Threshold 70%
                matches.append(self.name_index[result_name])

        return matches

    def _filter_by_criteria(self, indices: list, parsed_query: dict) -> list:
        """Filter wisata berdasarkan kriteria yang diparsing"""
        if not indices:
            return []

        filtered = indices.copy()

        # Filter berdasarkan lokasi
        if parsed_query['locations']:
            location_filtered = []
            for loc in parsed_query['locations']:
                loc_lower = loc.lower()
                # Cari di semua sinonim lokasi
                synonyms = preprocessor.location_synonyms.get(loc, [])
                all_terms = [loc_lower] + [s.lower() for s in synonyms]

                for term in all_terms:
                    if term in self.location_index:
                        location_filtered.extend(self.location_index[term])

            if location_filtered:
                filtered = [idx for idx in filtered if idx in set(location_filtered)]
            else:
                return []

        # Filter berdasarkan kategori
        if parsed_query['categories']:
            category_filtered = []
            for cat in parsed_query['categories']:
                cat_lower = cat.lower()
                # Cari di mapping kategori
                synonyms = preprocessor.category_mapping.get(cat, [])
                all_terms = [cat_lower] + [s.lower() for s in synonyms]

                for term in all_terms:
                    if term in self.category_index:
                        category_filtered.extend(self.category_index[term])

            if category_filtered:
                filtered = [idx for idx in filtered if idx in set(category_filtered)]
            else:
                return []

        # Filter berdasarkan harga
        if parsed_query['price_filters']:
            price_filtered = []
            for price_filter in parsed_query['price_filters']:
                for idx in filtered:
                    row = self.df.iloc[idx]
                    harga_min = row['harga_min']

                    if price_filter['type'] == 'free':
                        if harga_min == 0:
                            price_filtered.append(idx)
                    elif price_filter['type'] == 'cheap':
                        if 0 < harga_min <= 50000:
                            price_filtered.append(idx)
                    elif price_filter['type'] == 'expensive':
                        if harga_min >= 100000:
                            price_filtered.append(idx)
                    elif price_filter['type'] == 'max':
                        if harga_min <= price_filter.get('max', 0):
                            price_filtered.append(idx)
                    elif price_filter['type'] == 'min':
                        if harga_min >= price_filter.get('min', 0):
                            price_filtered.append(idx)
                    elif price_filter['type'] == 'range':
                        min_val = price_filter.get('min', 0)
                        max_val = price_filter.get('max', float('inf'))
                        if min_val <= harga_min <= max_val:
                            price_filtered.append(idx)

            if price_filtered:
                filtered = list(set(price_filtered))
            else:
                return []

        # Filter berdasarkan fasilitas (positif)
        if parsed_query['facility_filters']:
            facility_filtered = []
            for fac in parsed_query['facility_filters']:
                if fac in self.facility_index:
                    facility_filtered.extend(self.facility_index[fac])

            if facility_filtered:
                filtered = [idx for idx in filtered if idx in set(facility_filtered)]
            else:
                return []

        # Filter berdasarkan negasi (fasilitas yang tidak boleh ada)
        if parsed_query['negation_filters']:
            for neg_fac in parsed_query['negation_filters']:
                if neg_fac in self.facility_index:
                    # Hapus indeks yang memiliki fasilitas ini
                    filtered = [idx for idx in filtered if idx not in self.facility_index[neg_fac]]

        return filtered

    def _get_query_features(self, parsed_query: dict) -> np.ndarray:
        """Create feature vector from query for CBF"""
        # Kategori bin
        query_categories = parsed_query['categories']
        # Use mlb to transform categories
        if query_categories:
            query_kat = mlb.transform([query_categories])
        else:
            query_kat = np.zeros((1, len(mlb.classes_)))

        # Fasilitas bin
        query_fac = np.zeros(len(fac_cols))
        for fac in parsed_query['facility_filters']:
            if fac in fac_cols:
                query_fac[fac_cols.index(fac)] = 1
        for neg in parsed_query['negation_filters']:
            if neg in fac_cols:
                query_fac[fac_cols.index(neg)] = 0

        # Turunan features (zeros for now)
        query_tur = np.zeros(len(turunan_cols))

        # Harga features
        query_harga = np.zeros(4)  # avg, min, max, level_encoded
        if parsed_query['price_filters']:
            pf = parsed_query['price_filters'][0]
            if 'min' in pf and 'max' in pf:
                min_val = pf['min']
                max_val = pf['max'] if pf['max'] != float('inf') else 1e12  # Replace inf with large number
                query_harga[0] = (min_val + max_val) / 2
                query_harga[1] = min_val
                query_harga[2] = max_val
                # Map to harga_level encoding
                if min_val == 0:
                    query_harga[3] = 0  # gratis
                elif min_val <= 20000:
                    query_harga[3] = 1  # sangat_murah
                elif min_val <= 50000:
                    query_harga[3] = 2  # murah
                elif min_val <= 100000:
                    query_harga[3] = 3  # sedang
                elif min_val <= 300000:
                    query_harga[3] = 4  # mahal
                else:
                    query_harga[3] = 5  # sangat_mahal

        # TF-IDF features
        query_text = ' '.join(parsed_query['processed_tokens'])
        if self.tfidf_vectorizer:
            query_tfidf = self.tfidf_vectorizer.transform([query_text]).toarray()
        else:
            query_tfidf = np.zeros((1, self.tfidf_features.shape[1]))

        # Combine all features
        query_vec = np.hstack([
            query_kat[0],
            query_fac,
            query_tur,
            query_harga,
            query_tfidf[0]
        ])

        # Handle any remaining inf
        query_vec = np.nan_to_num(query_vec, nan=0.0, posinf=1e12, neginf=-1e12)

        return query_vec

    def recommend_to_cards(self, query: str, top_k: int = 5) -> list:
        """
        Rekomendasi utama yang mengembalikan list of card dict
        
        Args:
            query: Pertanyaan atau permintaan user
            top_k: Jumlah rekomendasi maksimum
        
        Returns:
            List of dictionary, setiap dictionary mewakili satu wisata
        """
        original_query = query
        query_lower = query.lower().strip()

        # Update stats (pertahankan dari VSCode)
        self.recommendation_stats['total_queries'] += 1
        self.query_history.append({
            'query': original_query,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'result_count': 0
        })

        # Parse query
        parsed_query = self.parser.parse_query(query)

        # Handle berdasarkan intent
        if parsed_query['intent'] == 'location':
            result = self._handle_location_query_to_cards(parsed_query)
        elif parsed_query['intent'] == 'price':
            result = self._handle_price_query_to_cards(parsed_query)
        elif parsed_query['intent'] == 'facility':
            result = self._handle_facility_query_to_cards(parsed_query, top_k)
        else:  # recommendation, search, unknown
            result = self._handle_recommendation_query_to_cards(parsed_query, top_k)

        # Update history
        self.query_history[-1]['result_count'] = len(result) if result else 0
        if result and not (len(result) == 1 and 'error' in result[0]):
            self.recommendation_stats['successful_queries'] += 1
        else:
            self.recommendation_stats['failed_queries'] += 1

        return result

    def _handle_location_query_to_cards(self, parsed_query: dict) -> list:
        """Handle query lokasi dan return cards"""
        if not parsed_query['target_name']:
            return [{"error": "Silakan sebutkan nama tempat yang ingin Anda cari."}]

        # Cari berdasarkan nama
        indices = self._find_by_name(parsed_query['target_name'])

        if not indices:
            return [{"error": f"Maaf, tidak ditemukan tempat '{parsed_query['target_name']}' dalam database."}]

        cards = []
        for idx in indices[:3]:  # Tampilkan max 3 hasil
            card = self._build_card_dict(idx)
            cards.append(card)

        return cards

    def _handle_price_query_to_cards(self, parsed_query: dict) -> list:
        """Handle query harga dan return cards"""
        # Cari tempat berdasarkan nama jika ada
        indices = []
        if parsed_query['target_name']:
            indices = self._find_by_name(parsed_query['target_name'])

        # Jika tidak ada nama spesifik, gunakan semua data
        if not indices:
            indices = list(range(len(self.df)))

        # Filter berdasarkan kriteria lainnya
        indices = self._filter_by_criteria(indices, parsed_query)

        if not indices:
            return [{"error": "Tidak ditemukan tempat yang sesuai dengan kriteria harga."}]

        # Ambil beberapa hasil
        indices = indices[:5]

        cards = []
        for idx in indices:
            card = self._build_card_dict(idx)
            cards.append(card)

        return cards

    def _handle_facility_query_to_cards(self, parsed_query: dict, top_k: int = 5) -> list:
        """Handle query fasilitas dan return cards"""
        # Mulai dengan semua data
        indices = list(range(len(self.df)))

        # Filter berdasarkan kriteria
        indices = self._filter_by_criteria(indices, parsed_query)

        if not indices:
            return [{"error": "Tidak ditemukan wisata yang sesuai dengan kriteria fasilitas."}]

        # Use similarity ranking
        query_vec = self._get_query_features(parsed_query)

        # Combine numeric and TF-IDF features for comparison
        combined_features = np.hstack([self.numeric_features, self.tfidf_features])

        # Calculate similarity only for filtered indices
        filtered_features = combined_features[indices]
        sim_scores = cosine_similarity([query_vec], filtered_features)[0]

        # Sort by similarity
        sorted_idx = np.argsort(sim_scores)[::-1][:top_k]
        sorted_indices = [indices[i] for i in sorted_idx]

        cards = []
        for i, idx in enumerate(sorted_indices, 1):
            card = self._build_card_dict(idx)
            card['rank'] = i
            card['relevansi'] = int(sim_scores[sorted_idx[i-1]] * 100)
            cards.append(card)

        return cards

    def _handle_recommendation_query_to_cards(self, parsed_query: dict, top_k: int = 5) -> list:
        """Handle query rekomendasi dan return cards"""
        # Mulai dengan semua data
        indices = list(range(len(self.df)))

        # Filter berdasarkan kriteria
        indices = self._filter_by_criteria(indices, parsed_query)

        if not indices:
            # Fallback: tampilkan wisata populer
            indices = list(range(len(self.df)))
            indices.sort(key=lambda x: self.df.iloc[x]['pop_score'], reverse=True)
            indices = indices[:top_k]

            cards = []
            for i, idx in enumerate(indices, 1):
                card = self._build_card_dict(idx)
                card['rank'] = i
                card['relevansi'] = 0
                cards.append(card)

            return cards

        # Calculate similarity
        query_vec = self._get_query_features(parsed_query)

        # Combine numeric and TF-IDF features for comparison
        combined_features = np.hstack([self.numeric_features, self.tfidf_features])
        filtered_features = combined_features[indices]

        # Calculate CBF similarity
        sim_scores = cosine_similarity([query_vec], filtered_features)[0]

        # If XGBoost model is available, combine scores
        if self.xgb_model is not None:
            try:
                # Get XGBoost predictions for filtered items
                xgb_features = self.numeric_features[indices]
                xgb_preds = self.xgb_model.predict(xgb_features)

                # Normalize both scores
                norm_sim = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min() + 1e-8)
                norm_xgb = (xgb_preds - xgb_preds.min()) / (xgb_preds.max() - xgb_preds.min() + 1e-8)

                # Combine scores
                combined_scores = 0.6 * norm_sim + 0.4 * norm_xgb

                # Sort by combined score
                sorted_idx = np.argsort(combined_scores)[::-1][:top_k]
            except:
                # Fallback to CBF only
                sorted_idx = np.argsort(sim_scores)[::-1][:top_k]
        else:
            # Use CBF only
            sorted_idx = np.argsort(sim_scores)[::-1][:top_k]

        sorted_indices = [indices[i] for i in sorted_idx]

        cards = []
        for i, idx in enumerate(sorted_indices, 1):
            card = self._build_card_dict(idx)
            card['rank'] = i
            card['relevansi'] = int(sim_scores[sorted_idx[i-1]] * 100)
            cards.append(card)

        return cards

    def _build_card_dict(self, idx):
        """Buat dictionary untuk card dari satu wisata (pertahankan dari VSCode)"""
        row = self.df.iloc[idx]
        
        nama = row['nama'] if 'nama' in row else 'Wisata Tanpa Nama'
        slug = slugify(str(nama))
        
        # Cek gambar dengan beberapa kemungkinan lokasi
        gambar_url = None
        gambar_paths = [
            f"static/wisatagambar/{slug}/{slug}.jpg",
            f"static/images/{slug}.jpg",
            f"static/{slug}.jpg"
        ]
        
        for path in gambar_paths:
            if os.path.exists(path):
                gambar_url = f"/{path}"
                break
        
        if not gambar_url:
            # Coba cari file gambar dengan nama yang mirip
            static_dir = "static"
            for root, dirs, files in os.walk(static_dir):
                for file in files:
                    if slug in file.lower() and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        gambar_url = f"/{os.path.join(root, file)}"
                        break
                if gambar_url:
                    break
        
        if not gambar_url:
            gambar_url = "/static/no-image.jpg"
        
        # Format harga
        harga_min = row.get('harga_min', 0)
        harga_max = row.get('harga_max', 0)
        
        if harga_min == 0:
            harga_text = "Gratis"
        else:
            try:
                harga_text = f"Rp{int(harga_min):,}"
                if harga_max > harga_min:
                    harga_text += f" - Rp{int(harga_max):,}"
            except:
                harga_text = "Harga tidak tersedia"
        
        # Format lokasi
        lokasi_str = "Lokasi tidak tersedia"
        if 'lokasi' in row and isinstance(row['lokasi'], dict):
            lokasi = row['lokasi']
            lokasi_parts = []
            if lokasi.get('kecamatan'):
                lokasi_parts.append(str(lokasi['kecamatan']))
            if lokasi.get('kabupaten_kota'):
                lokasi_parts.append(str(lokasi['kabupaten_kota']))
            if lokasi.get('provinsi'):
                lokasi_parts.append(str(lokasi['provinsi']))
            
            if lokasi_parts:
                lokasi_str = ", ".join(lokasi_parts)
        elif 'lokasi_str' in row:
            lokasi_str = row['lokasi_str'].title()
        
        # Dapatkan link maps jika ada
        maps_link = ""
        if 'lokasi' in row and isinstance(row['lokasi'], dict):
            maps_link = row['lokasi'].get('maps', '')
        
        # Dapatkan deskripsi
        deskripsi = row.get('deskripsi', 'Tidak ada deskripsi')
        if len(deskripsi) > 200:
            deskripsi = deskripsi[:200] + "..."
        
        # Dapatkan kategori
        kategori_str = ""
        if 'kategori' in row and isinstance(row['kategori'], list):
            kategori_list = row['kategori']
            if kategori_list:
                kategori_str = ", ".join([str(k).title() for k in kategori_list[:3]])
        
        # Buat card dictionary
        card = {
            "nama": str(nama),
            "rating": float(row.get('rating', 0)),
            "review": int(row.get('jumlah_review', 0)),
            "harga": harga_text,
            "lokasi": lokasi_str,
            "deskripsi": deskripsi,
            "kategori": kategori_str,
            "maps": maps_link,
            "gambar": gambar_url,
            "relevansi": 0  # Akan diisi nanti
        }
        
        # Tambahkan fitur khusus jika ada
        special_features = []
        if row.get('target_keluarga', 0) == 1:
            special_features.append("Keluarga")
        if row.get('spot_sunset', 0) == 1:
            special_features.append("Sunset")
        if row.get('aktivitas_air', 0) == 1:
            special_features.append("Aktivitas Air")
        if row.get('suasana_tenang', 0) == 1:
            special_features.append("Tenang")
        
        if special_features:
            card['fitur'] = ", ".join(special_features[:3])
        
        return card

# === INISIALISASI RECOMMENDER SYSTEM ===
recommender = ImprovedRecommender(df, parser, numeric_features, tfidf_matrix, model_xgb, tfidf)

# === SMART RECOMMEND → RETURN LIST OF CARD DICT ===
def smart_recommend(query: str, top_k: int = 5) -> list:
    """
    Fungsi utama untuk mendapatkan rekomendasi dalam bentuk list of card dict
    
    Args:
        query: Pertanyaan atau permintaan user
        top_k: Jumlah rekomendasi maksimum
    
    Returns:
        List of dictionary, setiap dictionary mewakili satu wisata
    """
    
    # Guardrail untuk query yang tidak relevan
    if any(word in query.lower() for word in ["ibukota","presiden", "bandung", "jakarta", 
                                              "matematika", "sejarah", "cuaca", "makanan"]):
        return [{"error": "Maaf, saya hanya merekomendasikan **wisata di Papua Barat (Sorong, Raja Ampat, Teminabuan, Maybrat)**."}]
    
    # Gunakan recommender yang sama dengan Colab
    try:
        cards = recommender.recommend_to_cards(query, top_k=top_k)
        
        # Jika tidak ada hasil
        if not cards:
            return [{"error": f"Maaf, tidak ditemukan wisata yang cocok dengan '{query}'.\n💡 Coba gunakan kata kunci yang lebih umum atau kurangi filter."}]
        
        return cards
        
    except Exception as e:
        return [{"error": f"Terjadi kesalahan dalam memproses rekomendasi: {str(e)}"}]

print("\n" + "=" * 60)
print("🚀 SISTEM REKOMENDASI WISATA SIAP DIGUNAKAN!")
print("=" * 60)
print("\nGunakan fungsi: smart_recommend('query anda', top_k=5)")
print("Contoh: smart_recommend('wisata murah di teminabuan')")
print("Contoh: smart_recommend('sungai di maybrat')")
print("Contoh: smart_recommend('dimana kali kaca')")