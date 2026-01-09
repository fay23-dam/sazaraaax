
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
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import nltk
from nltk.corpus import stopwords
from rapidfuzz import fuzz, process
import sys
import traceback
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import OpenAI

warnings.filterwarnings('ignore')

# Download required NLTK data if needed
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✅ NLTK data downloaded successfully")
except:
    print("⚠️ NLTK download skipped")

# === Konfigurasi OpenAI ===
from dotenv import load_dotenv
load_dotenv()

# LALU ubah baris inisialisasi OpenAI menjadi:
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Inisialisasi OpenAI client
def init_openai_client():
    """Inisialisasi OpenAI client dengan penanganan error"""
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-") == False:
        print("⚠️ OpenAI API key tidak valid atau tidak ditemukan")
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Test connection sederhana
        client.models.list()
        print("✅ OpenAI client berhasil diinisialisasi")
        return client
    except Exception as e:
        print(f"❌ Error initializing OpenAI: {e}")
        return None

# Global client
client = init_openai_client()
USE_LLM = client is not None

# === Slugify Function ===
def slugify(text):
    """Convert text to URL-friendly slug"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    return text

# === KELAS FixedPreprocessor (Sesuai Colab + Perbaikan Kategori) ===
class FixedPreprocessor:
    """Preprocessor dengan perbaikan untuk understanding yang lebih baik - SAMA SEPERTI COLAB"""
    def __init__(self):
        # Mapping lokasi dengan sinonim yang lebih akurat - SAMA SEPERTI COLAB
        self.location_synonyms = {
            'kota_sorong': ['kota sorong', 'sorong kota'],
            'sorong_selatan': ['sorong selatan', 'sorsel'],
            'raja_ampat': ['raja ampat', 'rajaampat'],
            'teminabuan': ['teminabuan', 'temi'],
            'maybrat': ['maybrat', 'ayamaru'],
            'maladumes': ['maladumes'],
            'klawasi': ['klawasi'],
            'misool': ['misool', 'misol'],
            'waigeo': ['waigeo'],
            'makbon': ['makbon'],
            'klamono': ['klamono'],
            'bal_bulol': ['bal bulol'],
            'love_lagoon': ['love lagoon', 'lovelagoon'],
            'telaga_bintang': ['telaga bintang'],
            'kali_biru': ['kali biru', 'blue river'],
            'wayag': ['wayag'],
            'pasir_timbul': ['pasir timbul'],
            'mansuar': ['mansuar'],
            'teluk_mayalibit': ['teluk mayalibit'],
            'waigeo_barat': ['waigeo barat'],
            'pianemo': ['pianemo']
        }
        # Mapping kategori dengan sinonim yang komprehensif - SAMA SEPERTI COLAB + PERBAIKAN
        self.category_mapping = {
            'pantai': ['pantai', 'beach', 'tanjung'],
            'sungai': ['sungai', 'kali', 'river'], # <-- Ditambahkan 'kali'
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
            'spot_sunset': ['spot sunset', 'sunset spot'], # <-- Ditambahkan
            'wisata_alam': ['wisata alam', 'nature'],
            'snorkeling': ['snorkeling', 'diving'],
            'geologi': ['geologi', 'geological'],
            'taman_rekreasi': ['taman rekreasi'],
            'resort': ['resort'],
            'eksklusif': ['eksklusif', 'ekslusif', 'exclusive', 'premium', 'mahal', 'mewah']
        }
        # Mapping untuk fasilitas - SAMA SEPERTI COLAB
        self.facility_keywords = {
            'penginapan': ['penginapan', 'hotel', 'villa', 'resort', 'cottage', 'homestay', 'kamar inap'],
            'kolam_renang': ['kolam renang', 'kolam_renang', 'swimming pool', 'pool', 'berenang', 'kolam'],
            'restoran': ['restoran', 'restaurant', 'warung', 'kafe', 'makanan'],
            'parkir': ['parkir', 'parking'],
            'toilet': ['toilet', 'wc', 'kamar mandi'],
            'wifi': ['wifi', 'internet']
        }

    def normalize_location(self, location_dict) -> dict:
        """Normalisasi lokasi dengan menyimpan semua data asli - SAMA SEPERTI COLAB"""
        if not location_dict:
            return {'kecamatan': '', 'kabupaten_kota': '', 'provinsi': '', 'maps': ''}
        kecamatan = location_dict.get('kecamatan', '') or ''
        kabupaten = location_dict.get('kabupaten', '') or location_dict.get('kabupaten_kota', '') or ''
        provinsi = location_dict.get('provinsi', '') or 'Papua Barat Daya'
        maps = location_dict.get('maps', '') or '' # Handle maps jika tidak ada di dataset asli

        kecamatan = kecamatan.strip().title()
        kabupaten = kabupaten.strip().title()
        provinsi = provinsi.strip().title()
        return {
            'kecamatan': kecamatan,
            'kabupaten_kota': kabupaten,
            'provinsi': provinsi,
            'maps': maps
        }

    def extract_categories(self, nama: str, deskripsi: str, existing_categories: list) -> list:
        """Ekstrak kategori dari data - SAMA SEPERTI COLAB"""
        categories = set()
        if existing_categories:
            if isinstance(existing_categories, str):
                existing_categories = [existing_categories]
            for cat in existing_categories:
                cat_lower = str(cat).lower().strip()
                categories.add(cat_lower)

        nama_lower = nama.lower()
        deskripsi_lower = (deskripsi or "").lower()
        combined_text = f"{nama_lower} {deskripsi_lower}"

        for cat_name, synonyms in self.category_mapping.items():
            for synonym in synonyms:
                if synonym in combined_text:
                    categories.add(cat_name)
                    break
        return list(categories)

    def extract_facilities(self, fasilitas_data, turunan_data=None) -> Dict:
        """Ekstrak fasilitas dengan parsing yang lebih baik - SAMA SEPERTI COLAB"""
        facilities = set()
        facility_flags = {
            'penginapan': 0, 'kolam_renang': 0, 'restoran': 0,
            'parkir': 0, 'toilet': 0, 'wifi': 0
        }

        # Cek dari turunan_data terlebih dahulu (prioritas tinggi) - SAMA SEPERTI COLAB
        if turunan_data:
            if turunan_data.get('penginapan', 0) == 1:
                facilities.add('penginapan')
                facility_flags['penginapan'] = 1
            if turunan_data.get('punya_kolam_renang', 0) == 1:
                facilities.add('kolam_renang')
                facility_flags['kolam_renang'] = 1

        # Cek dari fasilitas_data
        if fasilitas_data:
            if isinstance(fasilitas_data, list):
                fasilitas_str = ' '.join([str(f).lower() for f in fasilitas_data])
            else:
                fasilitas_str = str(fasilitas_data).lower()

            for fac_type, keywords in self.facility_keywords.items():
                # Skip jika sudah ditemukan dari turunan
                if fac_type in facilities:
                    continue
                for keyword in keywords:
                    if keyword in fasilitas_str:
                        facilities.add(fac_type)
                        facility_flags[fac_type] = 1
                        break
        return {'list': list(facilities), 'flags': facility_flags}

    def process_dataframe(self, raw_dict) -> pd.DataFrame:
        """Proses data menjadi DataFrame dengan parsing fasilitas yang lebih baik - SAMA SEPERTI COLAB"""
        records = []
        for key, item in raw_dict.items():
            record = item.copy()
            # Ambil fitur_turunan terlebih dahulu
            turunan = record.pop('fitur_turunan', {})
            # Update record dengan turunan
            record.update(turunan)

            lokasi = self.normalize_location(record.get('lokasi', {}))
            record['lokasi'] = lokasi

            categories = set(self.extract_categories(
                record.get('nama', ''),
                record.get('deskripsi', ''),
                record.get('kategori', [])
            ))
            if record.get('wisata_eksklusif', 0) == 1:
                categories.add('eksklusif')
            record['kategori'] = list(categories)

            # Ekstrak fasilitas dengan turunan_data
            facilities = self.extract_facilities(
                record.get('fasilitas', ''),
                turunan  # Kirim data turunan
            )
            record['fasilitas_list'] = facilities['list']
            record.update(facilities['flags'])

            try:
                harga_min = float(record.get('harga_min', 0) or 0)
                harga_max = float(record.get('harga_max', 0) or 0)
            except:
                harga_min = 0
                harga_max = 0
            record['harga_min'] = harga_min
            record['harga_max'] = harga_max

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

            record['rating'] = float(record.get('rating', 0)) if record.get('rating') not in [None, ''] else 0
            record['jumlah_review'] = int(record.get('jumlah_review', 0)) if record.get('jumlah_review') not in [None, ''] else 0
            records.append(record)

        df = pd.DataFrame(records)

        # Pop score yang lebih baik untuk XGBoost target - SAMA SEPERTI COLAB
        df['pop_score'] = df.apply(lambda x: x['rating'] * (1 + np.log1p(x['jumlah_review'])), axis=1)

        # Lokasi string untuk pencarian
        df['lokasi_str'] = df['lokasi'].apply(lambda x: f"{x['kabupaten_kota']}".lower())
        df['lokasi_detil'] = df['lokasi'].apply(lambda x: f"{x['kecamatan']} {x['kabupaten_kota']}".lower())

        # Tambahkan flag untuk kota sorong vs sorong selatan
        df['is_kota_sorong'] = df['lokasi'].apply(lambda x: 'kota sorong' in x['kabupaten_kota'].lower())
        df['is_sorong_selatan'] = df['lokasi'].apply(lambda x: 'sorong selatan' in x['kabupaten_kota'].lower())
        df['is_raja_ampat'] = df['lokasi'].apply(lambda x: 'raja ampat' in x['kabupaten_kota'].lower())

        # Pastikan semua kolom turunan ada, set ke 0 jika tidak ada
        turunan_cols = [
            'punya_kolam_renang', 'target_keluarga', 'is_pantai', 'spot_foto',
            'buka_24jam', 'wisata_edukasi', 'spot_sunset', 'aktivitas_air',
            'aksesibilitas', 'eksotis_papua', 'air_jernih', 'wisata_bahari',
            'suasana_tenang', 'wisata_eksklusif', 'wisata_ekstrem', 'eco_tourism',
            'keunikan', 'spot_panorama'
        ]
        for col in turunan_cols:
            if col not in df.columns:
                df[col] = 0 # Set default ke 0 jika tidak ditemukan

        return df

# === KELAS ImprovedQueryParser (Sesuai Colab + Perbaikan Lokasi & Kategori) ===
class ImprovedQueryParser:
    """Parser query yang lebih akurat - SAMA SEPERTI COLAB + Perbaikan Deteksi Lokasi & Kategori"""
    def __init__(self, preprocessor: FixedPreprocessor, raw_dataset: dict = None):
        self.preprocessor = preprocessor
        self.raw_dataset = raw_dataset or {}
        self.stop_words = set(stopwords.words('indonesian'))

    def parse_query(self, query: str) -> dict:
        """Parse query menjadi structured intent - SAMA SEPERTI COLAB + Perbaikan Deteksi Lokasi & Kategori"""
        query_lower = query.lower().strip()
        result = {
            'original_query': query,
            'locations': [],
            'categories': [],
            'price_filters': [],
            'facility_filters': [],
            'negation_filters': [],
            'intent': 'search'
        }

        # Deteksi intent
        if any(word in query_lower for word in ['dimana', 'di mana', 'lokasi', 'letak', 'ada dimana']):
            result['intent'] = 'location'
        elif any(word in query_lower for word in ['harga', 'biaya', 'berapa', 'rp']):
            result['intent'] = 'price'
        elif any(word in query_lower for word in ['fasilitas', 'ada', 'punya']):
            result['intent'] = 'facility'

        # PARSE LOKASI - SAMA SEPERTI COLAB + PERBAIKAN UNTUK LOKASI LAINNYA
        # Cek dulu lokasi eksplisit yang kompleks
        if 'sorong selatan' in query_lower or 'sorsel' in query_lower:
            result['locations'].append('sorong selatan')
        elif 'kota sorong' in query_lower or ('sorong' in query_lower and 'selatan' not in query_lower and 'maybrat' not in query_lower):
            # Tambahkan pengecualian untuk "sorong" jika muncul bersama "maybrat" (mungkin maksudnya di Maybrat)
            result['locations'].append('kota sorong')
        if 'raja ampat' in query_lower:
            result['locations'].append('raja ampat')

        # PARSE LOKASI LAINNYA MENGGUNAKAN SYNONYM MAP - PERBAIKAN UTAMA
        # Iterasi melalui semua pasangan lokasi dan sinonim dari preprocessor
        for loc_name, synonyms in self.preprocessor.location_synonyms.items():
            # Pastikan lokasi belum ditambahkan sebelumnya
            if loc_name not in result['locations']:
                for synonym in synonyms:
                    # Gunakan 'in' untuk pencocokan substring
                    if synonym in query_lower:
                        result['locations'].append(loc_name)
                        print(f"   📍 DETEKSI LOKASI DARI SYNONYM: '{synonym}' -> '{loc_name}'")
                        break # Hentikan loop synonyms untuk loc_name ini jika sudah ditemukan


        # PARSE KATEGORI - SAMA SEPERTI COLAB + PERBAIKAN UNTUK FRASA MULTI-KATA
        # Gunakan category_mapping dari preprocessor
        found_categories = set()
        for cat_name, synonyms in self.preprocessor.category_mapping.items():
            for synonym in synonyms:
                # Cek keberadaan sinonim dalam query
                if synonym in query_lower:
                    found_categories.add(cat_name)
                    break # Jika ditemukan satu sinonim, lanjut ke kategori berikutnya

        result['categories'] = list(found_categories)

        # PARSE HARGA - SAMA SEPERTI COLAB
        if 'gratis' in query_lower or 'free' in query_lower:
            result['price_filters'].append({'type': 'free', 'min': 0, 'max': 0})
        if 'murah' in query_lower:
            result['price_filters'].append({'type': 'murah', 'min': 0, 'max': 50000})
        if 'mahal' in query_lower:
            result['price_filters'].append({'type': 'mahal', 'min': 100000, 'max': float('inf')})

        # PARSE FASILITAS DENGAN LEBIH BAIK - SAMA SEPERTI COLAB
        facility_map = {
            'penginapan': ['penginapan', 'hotel', 'villa', 'resort', 'kamar inap', 'inap'],
            'kolam_renang': ['kolam renang', 'kolam_renang', 'swimming pool', 'pool', 'berenang', 'kolam'],
            'restoran': ['restoran', 'restaurant', 'warung', 'kafe', 'makan'],
            'parkir': ['parkir', 'parking'],
            'toilet': ['toilet', 'wc', 'kamar mandi'],
            'wifi': ['wifi', 'internet']
        }

        words = query_lower.split()
        for fac, keywords in facility_map.items():
            fac_found = False
            fac_negated = False
            for keyword in keywords:
                if keyword in query_lower:
                    start_idx = query_lower.find(keyword)
                    if start_idx > 0:
                        before_text = query_lower[:start_idx]
                        before_words = before_text.split()
                        check_words = before_words[-3:] if len(before_words) >= 3 else before_words
                        negasi_keywords = ['tanpa', 'tidak', 'no', 'belum']
                        if any(neg_word in check_words for neg_word in negasi_keywords):
                            fac_negated = True
                        else:
                            fac_found = True
                    else:
                        fac_found = True
                    break # Keluar dari loop keywords untuk fasilitas ini

            if fac_found:
                if fac not in result['facility_filters']:
                    result['facility_filters'].append(fac)
            elif fac_negated:
                if fac not in result['negation_filters']:
                    result['negation_filters'].append(fac)

        # Range harga spesifik - SAMA SEPERTI COLAB
        price_patterns = [
            (r'(\d+)\s*(?:ribu|rb|k)\s*sampai\s*(\d+)\s*(?:ribu|rb|k)', 'range'),
            (r'(\d+)\s*[-–]\s*(\d+)\s*(?:ribu|rb|k)', 'range'),
            (r'(\d+)\s*(?:ribu|rb|k)', 'single'),
        ]
        for pattern, price_type in price_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        try:
                            min_val = int(match[0]) * 1000
                            max_val = int(match[1]) * 1000
                            if min_val > max_val:
                                min_val, max_val = max_val, min_val
                            result['price_filters'].append({'type': 'range', 'min': min_val, 'max': max_val})
                        except:
                            pass
                    elif price_type == 'single':
                        try:
                            value = int(match) * 1000
                            margin = value * 0.3
                            result['price_filters'].append({'type': 'range', 'min': max(0, value - margin), 'max': value + margin})
                        except:
                            pass

        return result

# === KELAS EnhancedLLMQueryProcessor (Diperbarui - Perbaikan Fallback) ===
class EnhancedLLMQueryProcessor:
    """Processor query dengan LLM"""
    def __init__(self, raw_dataset: dict):
        self.client = client
        self.raw_dataset = raw_dataset
        self.USE_LLM = USE_LLM
        # Kata-kata sapaan
        self.greetings = ['halo', 'hai', 'hi', 'hello', 'assalamualaikum', 'pagi', 'siang', 'malam']
        # Kata-kata pertanyaan tentang bot
        self.about_bot = ['siapa kamu', 'kamu siapa', 'perkenalkan diri', 'apa itu', 'kamu itu', 'kamu adalah']
        # Kata-kata perintah umum
        self.general_questions = ['terima kasih', 'thanks', 'makasih', 'oke', 'okee', 'oke deh', 'baik', 'ya', 'iya']
        # Kata-kata untuk mencari tempat spesifik
        self.specific_place_keywords = ['dimana', 'di mana', 'lokasi', 'letak', 'fasilitas', 'apa saja', 'ada apa', 'berapa', 'harga', 'berapa harga', 'jam', 'operasional', 'info', 'informasi', 'tentang']

    def analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Analisis query menggunakan LLM"""
        if not self.client or not self.USE_LLM:
            return self.fallback_analysis(query)
        try:
            system_prompt = """Anda adalah analisis query untuk sistem rekomendasi wisata Papua Barat Daya.
Tugas Anda adalah menganalisis query pengguna dan menentukan dengan akurasi tinggi:
1. Apakah query adalah sapaan/perkenalan? (greeting)
2. Apakah query adalah pertanyaan tentang identitas atau fungsi bot? (about_bot)
3. Apakah query adalah ucapan terima kasih atau kata umum? (general)
4. Apakah query meminta informasi spesifik tentang tempat wisata tertentu (misalnya 'dimana Mooi Park', 'fasilitas di Tanjung Kasuari')? (specific_place_info)
5. Apakah query meminta rekomendasi wisata umum berdasarkan kriteria? (misalnya 'sungai di maybrat', 'pantai di misol', 'rekomendasi tempat wisata di sorsel') (recommendation)
6. Apakah query tidak dapat dikategorikan? (unknown)
Berikan jawaban dalam format JSON: {"query_type": "...", "confidence": float}"""
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0.2,
                max_tokens=200,
                response_format={ "type": "json_object" }
            )
            result_text = response.choices[0].message.content.strip()
            import json
            result_json = json.loads(result_text)
            query_type = result_json.get('query_type', 'unknown')
            confidence = result_json.get('confidence', 0.5)
            return {
                'is_greeting': query_type == 'greeting',
                'is_about_bot': query_type == 'about_bot',
                'is_general': query_type == 'general',
                'is_specific_place_info': query_type == 'specific_place_info',
                'query_type': query_type,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return self.fallback_analysis(query)

    def fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis tanpa LLM - DIPERBAIKI UNTUK MEMBEDAKAN SPESIFIK DAN REKOMENDASI"""
        query_lower = query.lower().strip()
        # Cek sapaan
        if any(greet in query_lower for greet in self.greetings):
            return {
                'is_greeting': True, 'is_about_bot': False, 'is_general': False,
                'is_specific_place_info': False, 'query_type': 'greeting', 'confidence': 0.9
            }
        # Cek tentang bot
        if any(about in query_lower for about in self.about_bot):
            return {
                'is_greeting': False, 'is_about_bot': True, 'is_general': False,
                'is_specific_place_info': False, 'query_type': 'about_bot', 'confidence': 0.9
            }
        # Cek umum
        if any(gen in query_lower for gen in self.general_questions):
            return {
                'is_greeting': False, 'is_about_bot': False, 'is_general': True,
                'is_specific_place_info': False, 'query_type': 'general', 'confidence': 0.9
            }
        # Cek kata kunci spesifik tempat dan coba temukan nama tempat EKSAK
        # Hanya anggap spesifik jika query hanya berisi kata kunci dan NAMA EKSAK tempat
        if any(word in query_lower for word in self.specific_place_keywords):
            # Jika query mengandung kata kunci spesifik dan NAMA EKSAK tempat, baru anggap specific_place_info
            for place_name in self.raw_dataset.keys():
                place_nama = self.raw_dataset[place_name].get('nama', '').lower()
                # Cocokkan nama eksak tempat dalam query
                if place_nama in query_lower or place_name.lower() in query_lower:
                    # Hanya jika query adalah nama tempat atau sangat dekat dengan nama tempat + kata kunci
                    # Misalnya: "dimana Mooi Park Sorong" cocok dengan "mooi park sorong"
                    # Tapi "sungai di teminabuan" TIDAK cocok dengan "mooi park sorong"
                    # Kita bisa gunakan panjang query sebagai filter tambahan
                    # Jika query jauh lebih panjang dari nama tempat, kemungkinan besar bukan permintaan info spesifik.
                    # Misalnya, jika panjang query > panjang nama tempat + ambang tertentu (misalnya 10 karakter)
                    # Maka ini kemungkinan besar adalah permintaan pencarian umum.
                    if len(query_lower) <= len(place_nama) + 10: # Ambang tambahan untuk kata kunci seperti 'dimana'
                        return {
                            'is_greeting': False, 'is_about_bot': False, 'is_general': False,
                            'is_specific_place_info': True, 'query_type': 'specific_place_info', 'confidence': 0.85
                        }
                    # Jika query lebih panjang dari ambang, mungkin itu adalah pencarian dengan kriteria.
                    # Misalnya "dimana tempat wisata dengan sunset di kota sorong?" -> ini rekomendasi.
                    # Kita tetap kembalikan unknown atau recommendation di sini, bukan specific_place_info.
                    # Untuk sementara, kita fokuskan hanya pada nama eksak yang pendek.
                    # Mungkin perlu logika LLM untuk kasus kompleks ini, jadi fallback ke unknown/recommendation.
                    # Kita abaikan nama yang ditemukan jika query terlalu panjang.
        # Cek kata kunci rekomendasi
        if any(word in query.lower() for word in ['rekomendasi', 'cari', 'tempat', 'wisata']):
            return {
                'is_greeting': False, 'is_about_bot': False, 'is_general': False,
                'is_specific_place_info': False, 'query_type': 'recommendation', 'confidence': 0.8
            }

        # Jika tidak cocok dengan yang di atas, dan mengandung kata kunci spesifik, mungkin adalah permintaan info umum
        if any(word in query_lower for word in self.specific_place_keywords):
            # Misalnya: "dimana sungai di teminabuan?" -> Ini adalah permintaan pencarian, bukan info spesifik tempat.
            # Kita asumsikan ini adalah permintaan rekomendasi berdasarkan kriteria.
            return {
                'is_greeting': False, 'is_about_bot': False, 'is_general': False,
                'is_specific_place_info': False, 'query_type': 'recommendation', 'confidence': 0.7 # Sedikit lebih rendah dari rekomendasi eksplisit
            }

        return {
            'is_greeting': False, 'is_about_bot': False, 'is_general': False,
            'is_specific_place_info': False, 'query_type': 'unknown', 'confidence': 0.5
        }

# === KELAS EnhancedRecommender (Diperbarui untuk Hybrid - SAMA SEPERTI COLAB) ===
class EnhancedRecommender:
    """Sistem rekomendasi dengan perbaikan filter lokasi dan hybrid ranking - SAMA SEPERTI COLAB"""
    def __init__(self, df: pd.DataFrame, parser: ImprovedQueryParser,
                 numeric_features, tfidf_features, xgb_model=None,
                 mlb=None, le_harga=None, tfidf_vectorizer=None):
        self.df = df
        self.parser = parser
        self.numeric_features = numeric_features
        self.tfidf_features = tfidf_features
        self.xgb_model = xgb_model
        self.mlb = mlb
        self.le_harga = le_harga
        self.tfidf_vectorizer = tfidf_vectorizer # Dibutuhkan untuk CBF
        # Kolom fasilitas
        self.fac_cols = ['penginapan', 'kolam_renang', 'restoran', 'parkir', 'toilet', 'wifi']
        # Kolom turunan
        self.turunan_cols = [
            'punya_kolam_renang', 'target_keluarga', 'is_pantai', 'spot_foto',
            'buka_24jam', 'wisata_edukasi', 'spot_sunset', 'aktivitas_air',
            'aksesibilitas', 'eksotis_papua', 'air_jernih', 'wisata_bahari',
            'suasana_tenang', 'wisata_eksklusif', 'wisata_ekstrem', 'eco_tourism',
            'keunikan', 'spot_panorama'
        ]

        # Gabungkan semua fitur untuk CBF similarity (seperti di Colab)
        self.cbf_features = np.hstack([
            self.mlb.transform(df['kategori'].apply(lambda x: x if isinstance(x, list) else [])),
            df[self.fac_cols].values.astype(np.float32),
            df[self.turunan_cols].values.astype(np.float32),
            df[['harga_avg', 'harga_min', 'harga_max', 'harga_level_encoded']].values.astype(np.float32),
            self.tfidf_features.astype(np.float32)
        ])
        self.cbf_features = np.nan_to_num(self.cbf_features, nan=0.0, posinf=1e12, neginf=-1e12)

        self._build_indexes()

    def _build_indexes(self):
        """Bangun indeks untuk pencarian cepat - SAMA SEPERTI COLAB"""
        self.name_index = {}
        self.category_index = {}
        self.location_index = {}
        self.facility_index = {}
        self.price_range_index = {}
        print("🔨 Building indexes...")
        for idx, row in self.df.iterrows():
            # Index nama
            name = row['nama'].lower()
            self.name_index[name] = idx
            # Index kategori
            for category in row['kategori']:
                if category not in self.category_index:
                    self.category_index[category] = []
                self.category_index[category].append(idx)
            # Index lokasi (kabupaten/kota)
            lokasi = row['lokasi']
            kabupaten = lokasi['kabupaten_kota'].lower()
            kecamatan = lokasi['kecamatan'].lower()
            # Index by kabupaten
            if kabupaten:
                if kabupaten not in self.location_index:
                    self.location_index[kabupaten] = []
                self.location_index[kabupaten].append(idx)
            # Index by kecamatan
            if kecamatan:
                if kecamatan not in self.location_index:
                    self.location_index[kecamatan] = []
                self.location_index[kecamatan].append(idx)
            # Index fasilitas
            for facility in ['penginapan', 'kolam_renang', 'restoran', 'parkir', 'toilet', 'wifi']:
                if row.get(facility, 0) == 1:
                    if facility not in self.facility_index:
                        self.facility_index[facility] = []
                    self.facility_index[facility].append(idx)
            # Index harga
            harga_min = row['harga_min']
            if harga_min == 0:
                self._add_to_price_index('gratis', idx)
            elif harga_min <= 20000:
                self._add_to_price_index('sangat_murah', idx)
            elif harga_min <= 50000:
                self._add_to_price_index('murah', idx)
            elif harga_min <= 100000:
                self._add_to_price_index('sedang', idx)
            elif harga_min <= 300000:
                self._add_to_price_index('mahal', idx)
            else:
                self._add_to_price_index('sangat_mahal', idx)
        print(f"   ✅ Name index: {len(self.name_index)} entries")
        print(f"   ✅ Category index: {len(self.category_index)} entries")
        print(f"   ✅ Location index: {len(self.location_index)} entries")
        print(f"   ✅ Facility index: {len(self.facility_index)} entries")

    def _add_to_price_index(self, price_level: str, idx: int):
        """Helper untuk menambahkan ke price index"""
        if price_level not in self.price_range_index:
            self.price_range_index[price_level] = []
        self.price_range_index[price_level].append(idx)

    def _create_query_vector(self, parsed_query: Dict) -> np.ndarray:
        """Buat vektor query untuk similarity calculation - SAMA SEPERTI COLAB"""
        # Default zero vector
        query_vector = np.zeros(self.cbf_features.shape[1])
        # Add category weights
        if parsed_query['categories']:
            for cat in parsed_query['categories']:
                if cat in self.mlb.classes_:
                    idx = list(self.mlb.classes_).index(cat)
                    query_vector[idx] = 1.0
        # Add facility weights (offset setelah kategori)
        fac_offset = len(self.mlb.classes_)
        fac_cols = ['penginapan', 'kolam_renang', 'restoran', 'parkir', 'toilet', 'wifi']
        for fac in parsed_query['facility_filters']:
            if fac in fac_cols:
                idx = fac_cols.index(fac)
                query_vector[fac_offset + idx] = 1.0
        # Normalize vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        return query_vector

    def _filter_by_location(self, indices: List[int], location: str) -> List[int]:
        """Filter berdasarkan lokasi dengan logika yang tepat - SAMA SEPERTI COLAB"""
        if not location or not indices:
            return []
        location_lower = location.lower()
        matched_indices = []
        # LOGIKA KHUSUS UNTUK SORONG SELATAN vs KOTA SORONG
        if location_lower == 'sorong selatan' or location_lower == 'sorsel':
            # HANYA wisata di Kabupaten Sorong Selatan
            for idx in indices:
                row = self.df.iloc[idx]
                kabupaten = row['lokasi']['kabupaten_kota'].lower()
                if 'sorong selatan' in kabupaten:
                    matched_indices.append(idx)
                    print(f"      ✓ {row['nama']} - {kabupaten}")

        elif location_lower == 'kota sorong':
            # HANYA wisata di Kota Sorong
            for idx in indices:
                row = self.df.iloc[idx]
                kabupaten = row['lokasi']['kabupaten_kota'].lower()
                if 'kota sorong' in kabupaten:
                    matched_indices.append(idx)
                    print(f"      ✓ {row['nama']} - {kabupaten}")
        else:
            # Untuk lokasi lain, gunakan index biasa
            if location_lower in self.location_index:
                matched_indices = [idx for idx in indices if idx in self.location_index[location_lower]]
            else:
                # Cari partial match
                for loc_key in self.location_index.keys():
                    if location_lower in loc_key or loc_key in location_lower:
                        matched_indices.extend([idx for idx in indices if idx in self.location_index[loc_key]])
                matched_indices = list(set(matched_indices))
        print(f"      🔍 Found {len(matched_indices)} matches for location '{location_lower}'")
        return matched_indices

    def recommend(self, parsed_query: Dict, top_k: int = 10) -> List[int]:
        """Rekomendasi berdasarkan query yang sudah diparsing - SAMA SEPERTI COLAB"""
        # Mulai dari semua wisata
        indices = list(range(len(self.df)))
        print(f"🔍 Processing query: '{parsed_query['original_query']}'")
        print(f"   📍 Locations: {parsed_query['locations']}")
        print(f"   🏷️ Categories: {parsed_query['categories']}")
        print(f"   🏪 Facility filters: {parsed_query['facility_filters']}")

        # Filter bertahap dengan prioritas - SAMA SEPERTI COLAB
        # 1. Filter lokasi
        if parsed_query['locations']:
            location_filtered = []
            for location in parsed_query['locations']:
                filtered = self._filter_by_location(indices, location)
                location_filtered.extend(filtered)
            if location_filtered:
                indices = list(set(location_filtered))

        # 2. Filter kategori
        if parsed_query['categories'] and indices:
            category_filtered = []
            for category in parsed_query['categories']:
                if category in self.category_index:
                    category_filtered.extend([idx for idx in self.category_index[category] if idx in indices])
            if category_filtered:
                indices = list(set(category_filtered))

        # 3. Filter harga
        if parsed_query['price_filters'] and indices:
            price_filtered = []
            for price_filter in parsed_query['price_filters']:
                for idx in indices:
                    row = self.df.iloc[idx]
                    harga_min = row['harga_min']
                    if price_filter['type'] == 'free':
                        if harga_min == 0:
                            price_filtered.append(idx)
                    elif price_filter['type'] == 'murah':
                        if 0 < harga_min <= 50000:
                            price_filtered.append(idx)
                    elif price_filter['type'] == 'mahal':
                        if harga_min >= 100000:
                            price_filtered.append(idx)
                    elif price_filter['type'] == 'range':
                        min_val = price_filter.get('min', 0)
                        max_val = price_filter.get('max', float('inf'))
                        if max_val == float('inf'):
                            max_val = 1e12
                        if min_val <= harga_min <= max_val:
                            price_filtered.append(idx)
            if price_filtered:
                indices = list(set(price_filtered))

        # 4. Filter fasilitas
        if parsed_query['facility_filters'] and indices:
            facility_filtered = []
            for facility in parsed_query['facility_filters']:
                if facility in self.facility_index:
                    facility_filtered.extend([idx for idx in self.facility_index[facility] if idx in indices])
            if facility_filtered:
                indices = list(set(facility_filtered))

        # 5. Filter negasi
        if parsed_query['negation_filters'] and indices:
            for neg_fac in parsed_query['negation_filters']:
                if neg_fac in self.facility_index:
                    indices = [idx for idx in indices if idx not in self.facility_index[neg_fac]]

        print(f"   📊 After filtering: {len(indices)} results found")

        # Jika tidak ada hasil setelah filter
        if not indices:
            print("   ❌ Tidak ditemukan wisata yang sesuai.")
            # Fallback: kembalikan wisata berdasarkan pop_score jika tidak ada filter yang cocok
            all_indices = list(range(len(self.df)))
            all_indices.sort(key=lambda x: self.df.iloc[x]['pop_score'], reverse=True)
            return all_indices[:top_k]

        # HYBRID RANKING: CBF + XGBoost - SAMA SEPERTI COLAB
        if len(indices) > 1:
            print(f"   🤖 Menggunakan Hybrid ranking (CBF + XGBoost)")
            # 1. Hitung similarity score (CBF)
            query_vector = self._create_query_vector(parsed_query)
            filtered_cbf_features = self.cbf_features[indices]
            # Hitung cosine similarity
            similarity_scores = cosine_similarity([query_vector], filtered_cbf_features)[0]

            # 2. Hitung XGBoost score
            xgb_scores = self.xgb_model.predict(self.numeric_features[indices]) # Gunakan numeric_features untuk XGBoost

            # Normalize scores
            similarity_scores_norm = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min() + 1e-8)
            xgb_scores_norm = (xgb_scores - xgb_scores.min()) / (xgb_scores.max() - xgb_scores.min() + 1e-8)

            # Combine scores dengan weighting (70% CBF, 30% XGBoost)
            combined_scores = 0.7 * similarity_scores_norm + 0.3 * xgb_scores_norm

            # Sort berdasarkan combined scores
            sorted_indices_within_filtered = np.argsort(combined_scores)[::-1]
            indices = [indices[i] for i in sorted_indices_within_filtered[:top_k]]
        else:
            indices = indices[:top_k]

        return indices

# === KELAS SmartTourismRecommender (Diperbarui) ===
class SmartTourismRecommender:
    """Sistem rekomendasi wisata utama - DIPERBARUI"""
    def __init__(self, model_dir: str = 'model_wisata_recommender_x3'):
        """Inisialisasi sistem rekomendasi"""
        print("\n" + "="*60)
        print("🚀 SMART TOURISM RECOMMENDER SYSTEM (Updated to match Colab Hybrid + Typo Handling + Query Type Fix)")
        print("="*60)
        print(f"📊 Mode LLM: {'AKTIF' if USE_LLM else 'NON-AKTIF'}")
        self.model_dir = model_dir
        self.load_models()
        self.load_dataset()
        self.initialize_components()
        print("✅ Sistem rekomendasi siap digunakan!")
        print("="*60)

    def load_models(self):
        """Memuat semua model yang diperlukan - DIPERBARUI UNTUK COCOK DENGAN COLAB PIPELINE"""
        try:
            print("📦 Memuat model...")
            # Muat model XGBoost
            xgb_model_path = os.path.join(self.model_dir, 'xgb_model.json')
            if os.path.exists(xgb_model_path):
                self.model_xgb = xgb.XGBRegressor()
                self.model_xgb.load_model(xgb_model_path)
                print("   ✅ XGBoost model loaded")
            else:
                self.model_xgb = None
                print("   ⚠️  XGBoost model tidak ditemukan, menggunakan pop_score fallback")

            # Muat TF-IDF Vectorizer
            tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
            self.tfidf = joblib.load(tfidf_path)
            print("   ✅ TF-IDF vectorizer loaded")

            # Muat MultiLabelBinarizer
            mlb_path = os.path.join(self.model_dir, 'mlb_kategori.pkl')
            self.mlb = joblib.load(mlb_path)
            print("   ✅ MultiLabelBinarizer loaded")

            # Muat LabelEncoder untuk harga
            le_harga_path = os.path.join(self.model_dir, 'le_harga.pkl')
            self.le_harga = joblib.load(le_harga_path)
            print("   ✅ Harga encoder loaded")

            # Muat processed dataframe
            df_path = os.path.join(self.model_dir, 'processed_df.pkl')
            self.df = pd.read_pickle(df_path)
            print(f"   ✅ DataFrame loaded: {len(self.df)} rows")

            # Muat numeric features (ini adalah fitur gabungan untuk XGBoost dari pipeline Colab)
            numeric_path = os.path.join(self.model_dir, 'all_features.npy') # Nama file dari Colab
            self.numeric_features = np.load(numeric_path)
            print(f"   ✅ Numeric features (XGBoost): {self.numeric_features.shape}")

            # Muat TF-IDF features (ini adalah fitur deskripsi untuk CBF dari pipeline Colab)
            tfidf_features_path = os.path.join(self.model_dir, 'tfidf_features.npy')
            self.tfidf_features = np.load(tfidf_features_path)
            print(f"   ✅ TF-IDF features (CBF): {self.tfidf_features.shape}")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            traceback.print_exc()
            raise

    def load_dataset(self):
        """Memuat dataset asli"""
        try:
            dataset_path = 'dataset.json'
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    self.raw_dataset = json.load(f)
                print(f"✅ Dataset loaded: {len(self.raw_dataset)} tempat wisata")
            else:
                print("⚠️ Dataset tidak ditemukan")
                self.raw_dataset = {}
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self.raw_dataset = {}

    def initialize_components(self):
        """Inisialisasi komponen sistem - DIPERBARUI"""
        print("🔧 Inisialisasi komponen...")
        # Inisialisasi preprocessor
        self.preprocessor = FixedPreprocessor()
        # Inisialisasi parser
        self.parser = ImprovedQueryParser(self.preprocessor, self.raw_dataset)
        # Inisialisasi LLM processor
        self.llm_processor = EnhancedLLMQueryProcessor(self.raw_dataset)
        # Inisialisasi EnhancedRecommender dengan Hybrid
        self.enhanced_recommender = EnhancedRecommender(
            df=self.df,
            parser=self.parser,
            numeric_features=self.numeric_features, # Fitur untuk XGBoost
            tfidf_features=self.tfidf_features, # Fitur untuk CBF
            xgb_model=self.model_xgb,
            mlb=self.mlb,
            le_harga=self.le_harga,
            tfidf_vectorizer=self.tfidf # Tambahkan TF-IDF untuk CBF
        )
        # Membuat mapping nama ke raw data
        self.create_name_mapping()
        print("   ✅ Semua komponen siap")

    def create_name_mapping(self):
        """Membuat mapping nama tempat ke data raw"""
        self.name_to_raw_data = {}
        for place in self.raw_dataset.values():
            name = place.get('nama', '')
            if name:
                self.name_to_raw_data[name] = place

    def format_recommendation(self, row: pd.Series) -> dict:
        """Format rekomendasi dengan mengambil data dari raw_dataset (seperti specific_place_info)"""
        nama = row['nama']
        
        # Cari data asli berdasarkan nama
        place_data = self.name_to_raw_data.get(nama)
        if not place_data:
            # Fallback jika tidak ditemukan (seharusnya tidak terjadi)
            print(f"⚠️ Warning: {nama} tidak ditemukan di raw dataset")
            return {
                "nama": str(nama),
                "rating": 0,
                "review": 0,
                "harga": "Tidak tersedia",
                "fasilitas": "",
                "lokasi": "",
                "deskripsi": "",
                "kategori": "",
                "maps": "",
                "gambar": "/static/no-image.jpg",
                "harga_min": 0,
                "harga_max": 0
            }
        
        # Gunakan fungsi yang sudah ada untuk format konsisten
        return self.format_recommendation_from_raw(place_data)

    def format_recommendation_from_raw(self, place_data: dict) -> dict:
        """Format data tempat dari raw JSON ke dalam format kartu rekomendasi"""
        nama = place_data.get('nama', 'Nama Tidak Tersedia')
        slug = slugify(str(nama))
        # Cari gambar
        gambar_url = self.find_image(slug)
        # Format harga
        harga_min = place_data.get('harga_min', 0)
        harga_max = place_data.get('harga_max', 0)
        if harga_min == 0 and harga_max == 0:
            harga_text = "Gratis"
        else:
            try:
                harga_text = f"Rp{int(harga_min):,}"
                if harga_max > harga_min and harga_max != 0:
                    harga_text += f" - Rp{int(harga_max):,}"
                elif harga_max == 0:
                    # Jika hanya harga_min yang ada
                    pass
            except (ValueError, TypeError):
                harga_text = "Harga tidak tersedia"
        # Format lokasi
        lokasi_data = place_data.get('lokasi', {})
        lokasi_str = f"{lokasi_data.get('kecamatan', 'Kecamatan Tidak Tersedia')}, {lokasi_data.get('kabupaten_kota', 'Kabupaten/Kota Tidak Tersedia')}"
        maps_link = lokasi_data.get('maps', '')
        # Dapatkan deskripsi
        deskripsi = place_data.get('deskripsi', 'Tidak ada deskripsi')
        if len(deskripsi) > 200:
            deskripsi = deskripsi[:200] + "..."
        # Dapatkan kategori (ambil dari raw data)
        kategori_list = place_data.get('kategori', [])
        kategori_str = ", ".join([str(k).title() for k in kategori_list[:3]]) if kategori_list else "Kategori Tidak Tersedia"
        # Dapatkan fasilitas (ambil dari raw data)
        fasilitas_list = place_data.get('fasilitas', [])
        fasilitas_str = ", ".join([str(f).replace('_', ' ').title() for f in fasilitas_list[:5]]) if fasilitas_list else "Fasilitas Tidak Tersedia"
        # Buat card dictionary
        card = {
            "nama": str(nama),
            "rating": float(place_data.get('rating', 0)),
            "review": int(place_data.get('jumlah_review', 0)),
            "harga": harga_text,
            "fasilitas": fasilitas_str,
            "lokasi": lokasi_str,
            "deskripsi": deskripsi,
            "kategori": kategori_str,
            "maps": maps_link,
            "gambar": gambar_url,
            "harga_min": float(harga_min),
            "harga_max": float(harga_max),
        }
        # Tambahkan fitur khusus dari fitur_turunan jika ada
        turunan = place_data.get('fitur_turunan', {})
        special_features = []
        if turunan.get('target_keluarga', 0) == 1:
            special_features.append("Keluarga")
        if turunan.get('spot_sunset', 0) == 1:
            special_features.append("Sunset")
        if turunan.get('aktivitas_air', 0) == 1:
            special_features.append("Aktivitas Air")
        if turunan.get('wisata_eksklusif', 0) == 1:
            special_features.append("Eksklusif")
        if special_features:
            card['fitur'] = ", ".join(special_features[:3])

        return card

    def find_image(self, slug: str) -> str:
        """Cari gambar berdasarkan slug"""
        # Cek berbagai kemungkinan path
        gambar_paths = [
            f"static/{slug}/{slug}.jpg",
            f"static/images/{slug}.jpg",
            f"static/{slug}.jpg",
            f"static/no-image.jpg"
        ]
        for path in gambar_paths:
            if os.path.exists(path.replace('/', '\\')):
                return f"/{path}"
        return "/static/no-image.jpg"

    def _find_place_by_name_with_typo_handling(self, query: str) -> Optional[Dict]:
        """Cari tempat wisata berdasarkan nama dari query, dengan penanganan typo - SAMA SEPERTI COLAB"""
        query_lower = query.lower().strip()

        # 1. Cek apakah query mengandung nama tempat dari raw_dataset secara eksak
        for place_id, place_data in self.raw_dataset.items():
            place_nama = place_data.get('nama', '').lower()
            if place_nama in query_lower or place_id.lower() in query_lower:
                return place_data

        # 2. Jika tidak ditemukan eksak, coba fuzzy match (opsional)
        # Ambil semua nama tempat
        all_place_names = [data['nama'] for data in self.raw_dataset.values()]
        all_place_ids = list(self.raw_dataset.keys())

        # Gunakan rapidfuzz untuk fuzzy matching - SAMA SEPERTI COLAB
        # Cari nama yang paling mirip
        best_match_nama, score_nama, _ = process.extractOne(query_lower, [name.lower() for name in all_place_names], scorer=fuzz.partial_ratio)
        best_match_id, score_id, _ = process.extractOne(query_lower, all_place_ids, scorer=fuzz.partial_ratio)

        # Pilih yang skornya lebih tinggi
        best_match = None
        best_score = 0
        if score_nama > best_score:
            best_match = best_match_nama
            best_score = score_nama
        if score_id > best_score:
            best_match = self.raw_dataset[best_match_id].get('nama', '')
            best_score = score_id

        # 3. Gunakan LLM (jika aktif) untuk analisis tambahan
        # Perbaikan: Akses client dan USE_LLM dari self.llm_processor
        if self.llm_processor.client and self.llm_processor.USE_LLM:
            try:
                system_prompt = f"""Anda adalah asisten pencarian nama tempat wisata.
Dataset berisi nama-nama tempat wisata berikut: {', '.join(all_place_names[:10])}... (dan {len(all_place_names)} lainnya).
User menanyakan '{query}'. Apakah ada nama tempat wisata dalam dataset yang mirip atau merupakan typo dari query tersebut?
Jika ya, sebutkan nama aslinya. Jika tidak, jawab 'Tidak ditemukan'. Hanya jawab nama tempatnya saja atau 'Tidak ditemukan'."""
                response = self.llm_processor.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Query: {query}"}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                llm_guess = response.choices[0].message.content.strip()
                if llm_guess.lower() != "tidak ditemukan":
                    # Cocokkan jawaban LLM dengan nama asli
                    for place_data in self.raw_dataset.values():
                        if place_data.get('nama', '').lower() == llm_guess.lower():
                            return place_data
                    # Jika tidak cocok secara eksak, lakukan fuzzy match lagi terhadap jawaban LLM
                    llm_match_nama, llm_score, _ = process.extractOne(llm_guess.lower(), [name.lower() for name in all_place_names], scorer=fuzz.partial_ratio)
                    if llm_score > best_score:
                        best_match = llm_match_nama
                        best_score = llm_score
            except Exception as e:
                print(f"   ⚠️ Error calling LLM for typo correction: {e}")
        if best_match and best_score >= 60: # Naikkan threshold untuk mencegah false positive
            for place_data in self.raw_dataset.values():
                if place_data.get('nama', '').lower() == best_match.lower():
                    print(f"   📝 Typo diperbaiki: '{query}' -> '{place_data['nama']}' (Score: {best_score})")
                    return place_data

        return None # Tidak ditemukan


    def process_user_query(self, query: str, top_k: int = 10) -> dict:
        """Proses query pengguna - DIPERBARUI"""
        print(f"\n📝 QUERY: {query}")
        try:
            # Step 1: Analisis query dengan LLM
            query_analysis = self.llm_processor.analyze_query_with_llm(query)
            print(f"🔍 Analisis LLM: {query_analysis['query_type']} (Conf: {query_analysis['confidence']:.2f})")

            # Step 2: Handle berdasarkan tipe query dari LLM
            query_type = query_analysis['query_type']

            # --- Tambahkan kondisi untuk specific_place_info ---
            if query_type in ['greeting']:
                chat_response = " Halo! Saya adalah asisten rekomendasi wisata Papua Barat Daya. Saya bisa membantu Anda menemukan tempat wisata yang sesuai dengan minat Anda. Silakan tanyakan wisata seperti apa yang Anda cari!"
                return {
                    "success": True,
                    "message": "Sapaan",
                    "recommendations": [],
                    "llm_response": chat_response,
                    "count": 0,
                    "mode": "greeting"
                }
            elif query_type in ['about_bot']:
                chat_response = " Saya adalah sistem rekomendasi wisata yang dirancang khusus untuk membantu Anda menemukan tempat-tempat wisata menarik di Papua Barat Daya berdasarkan preferensi Anda seperti lokasi, kategori, fasilitas, dan harga."
                return {
                    "success": True,
                    "message": "Informasi tentang bot",
                    "recommendations": [],
                    "llm_response": chat_response,
                    "count": 0,
                    "mode": "about_bot"
                }
            elif query_type in ['general']:
                chat_response = " Sama-sama! Jika Anda ingin mencari wisata, silakan beri tahu saya kriterianya."
                return {
                    "success": True,
                    "message": "Respons umum",
                    "recommendations": [],
                    "llm_response": chat_response,
                    "count": 0,
                    "mode": "general"
                }
            elif query_type in ['specific_place_info']:
                # Cari tempat wisata berdasarkan nama dalam query, dengan penanganan typo
                found_place_data = self._find_place_by_name_with_typo_handling(query)
                if found_place_data:
                    # Format sebagai satu kartu rekomendasi
                    card = self.format_recommendation_from_raw(found_place_data)
                    place_nama = found_place_data.get('nama', 'Tempat Wisata')
                    chat_response = f"Berikut informasi tentang **{place_nama}**:"
                    return {
                        "success": True,
                        "message": f"Info untuk {place_nama}",
                        "recommendations": [card], # Kembalikan sebagai list dengan satu item
                        "llm_response": chat_response,
                        "count": 1,
                        "mode": "specific_place_info"
                    }
                else:
                    # Jika nama tempat dikenali tetapi tidak ditemukan
                    chat_response = f"Maaf, saya tidak dapat menemukan informasi untuk tempat yang Anda maksud ('{query}')."
                    return {
                        "success": False,
                        "message": "Tempat tidak ditemukan",
                        "recommendations": [],
                        "llm_response": chat_response,
                        "count": 0,
                        "mode": "not_found_specific"
                    }
            elif query_type == 'recommendation':
                # Step 3: Parse query untuk ekstraksi fitur hanya jika ini adalah rekomendasi umum
                parsed_query = self.parser.parse_query(query)
                # Gunakan EnhancedRecommender untuk mendapatkan rekomendasi (Hybrid)
                print(f"\n🔎 Mencari rekomendasi umum dengan Hybrid System...")
                recommended_indices = self.enhanced_recommender.recommend(parsed_query, top_k=top_k)

                # Format rekomendasi
                recommendations = []
                for idx in recommended_indices:
                    rec = self.format_recommendation(self.df.iloc[idx])
                    recommendations.append(rec)

                if not recommendations:
                    chat_response = "Maaf, tidak ditemukan wisata yang sesuai dengan kriteria Anda."
                    return {
                        "success": False,
                        "message": "Tidak ditemukan",
                        "recommendations": [],
                        "llm_response": chat_response,
                        "count": 0,
                        "mode": "no_results"
                    }

                # Generate response
                if parsed_query['locations']:
                    locations = parsed_query['locations']
                    location_str = ", ".join(locations).replace('_', ' ').title()
                    if parsed_query['categories']:
                        categories_str = ", ".join(parsed_query['categories']).replace('_', ' ').title()
                        chat_response = f"Saya menemukan {len(recommendations)} wisata {categories_str} di {location_str}:"
                    else:
                        chat_response = f"Saya menemukan {len(recommendations)} wisata di {location_str}:"
                elif parsed_query['categories']:
                    categories_str = ", ".join(parsed_query['categories']).replace('_', ' ').title()
                    chat_response = f"Saya menemukan {len(recommendations)} wisata {categories_str}:"
                else:
                    chat_response = f"Saya menemukan {len(recommendations)} rekomendasi wisata untuk Anda:"

                return {
                    "success": True,
                    "message": "Rekomendasi berhasil",
                    "recommendations": recommendations,
                    "llm_response": chat_response,
                    "count": len(recommendations),
                    "mode": "recommendation"
                }
            else: # query_type == 'unknown'
                # Untuk query yang benar-benar tidak dikenali
                chat_response = "Maaf, saya tidak mengerti. Saya bisa membantu Anda mencari wisata di Papua Barat Daya. Anda bisa bertanya seperti 'rekomendasi wisata di Sorong' atau 'dimana Mooi Park'."
                return {
                    "success": False,
                    "message": "Query tidak dikenali",
                    "recommendations": [],
                    "llm_response": chat_response,
                    "mode": "unknown"
                }
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "recommendations": [],
                "llm_response": "Maaf, terjadi kesalahan. Silakan coba lagi.",
                "mode": "error"
            }

# ============================================================
# FUNGSI UTAMA UNTUK INTEGRASI
# ============================================================
# Global recommender instance
_recommender_instance = None

def get_recommender():
    """Dapatkan instance recommender (singleton pattern)"""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = SmartTourismRecommender()
    return _recommender_instance

def smart_recommend(query: str, top_k: int = 10) -> list:
    """
    Fungsi utama untuk mendapatkan rekomendasi
    Args:
        query: Pertanyaan user
        top_k: Jumlah rekomendasi maksimum
    Returns:
        List of recommendation cards
    """
    try:
        recommender = get_recommender()
        result = recommender.process_user_query(query, top_k=top_k)
        return result['recommendations']
    except Exception as e:
        print(f"Error in smart_recommend: {e}")
        return []

# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING SMART TOURISM RECOMMENDER (Updated)")
    print("="*60)
    # Inisialisasi sistem
    recommender = SmartTourismRecommender()
    # Test queries
    test_queries = [
        "rekomendasi wisata di sorong selatan",
        "wisata di kota sorong",
        "wisata murah di sorong",
        "pantai di kota sorong",
        "sungai di maybrat",
        "kali di teminabuan", # <-- Harus masuk ke recommendation
        "wisata eksklusif di raja ampat",
        "wisata dengan penginapan di sorong",
        "wisata dengan kolam renang di kota sorong",
        # Test typo
        "dimana lenmakana", # Typo untuk lenmakana
        "dimana lemnakana", # Typo untuk lenmakana
        "dimana mooi park sorong", # Nama eksak
        "dimana tajung kasuari", # Typo untuk tanjung kasuari
        # Test sapaan dan umum
        "halo",
        "siapa kamu",
        "terima kasih",
        # Test kategori multi-kata
        "rekomendasi spot sunset di kota sorong", # <-- Harus mendeteksi kategori spot_sunset
    ]
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"🔍 QUERY: '{query}'")
        print("-"*30)
        result = recommender.process_user_query(query, top_k=3)
        print(f"📊 Mode response: {result['mode']}")
        print(f"📊 Jumlah rekomendasi: {result['count']}")
        if result['success'] and result['recommendations']:
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"  {i}. {rec['nama']}")
                print(f"     Lokasi: {rec['lokasi']}")
                print(f"     Harga: {rec['harga']}")
                print(f"     Rating: {rec['rating']}")
        elif 'llm_response' in result:
            print(f"💬 Respons: {result['llm_response']}")
    print("\n" + "="*60)
    print("✅ TESTING SELESAI")
    print("="*60)