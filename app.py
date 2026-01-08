from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import traceback
import json

app = Flask(__name__, template_folder='templates', static_folder='static')

print("=" * 60)
print("🚀 Starting Flask Application...")
print("=" * 60)

# Periksa apakah folder model ada
model_dir = "model_wisata_recommender_x3"
if not os.path.exists(model_dir):
    print(f"❌ ERROR: Folder model '{model_dir}' tidak ditemukan!")
    exit(1)

print("📦 Loading model & data...")
try:
    from recommender import SmartTourismRecommender
    print("✅ Komponen berhasil diimport!")
    
    # Inisialisasi recommender
    recommender = SmartTourismRecommender(model_dir)
    print("✅ Sistem rekomendasi siap digunakan!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    traceback.print_exc()
    exit(1)

# =============================================
# ROUTES
# =============================================

@app.route('/')
def home():
    """Halaman utama (landing page)"""
    return render_template('home.html')

@app.route('/chat')
def chat():
    """Halaman chat/rekomendasi"""
    return render_template('index.html')

@app.route('/index')
def index_redirect():
    """Redirect untuk kompatibilitas"""
    return redirect(url_for('chat'))

@app.route('/index.html')
def index_html():
    """Redirect untuk file langsung"""
    return redirect(url_for('chat'))

# =============================================
# API ENDPOINTS
# =============================================

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint untuk mendapatkan rekomendasi"""
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    
    if not user_msg:
        return jsonify({"cards": []})
    
    try:
        print(f"\n💬 Permintaan dari user: '{user_msg}'")
        
        # Proses query (tanpa parameter use_llm)
        result = recommender.process_user_query(user_msg, top_k=5)
        
        response_data = {
            "success": result.get('success', False),
            "cards": result.get('recommendations', []),
            "llm_response": result.get('llm_response', ''),
            "show_cards_header": len(result.get('recommendations', [])) > 0,
            "mode": result.get('mode', 'unknown')
        }
        
        # Jika ada error
        if not result.get('success'):
            error_msg = result.get('llm_response', 'Maaf, terjadi kesalahan.')
            response_data["error"] = error_msg
        
        # Log untuk debugging
        print(f"📊 Mode response: {result.get('mode', 'unknown')}")
        print(f"📊 Jumlah rekomendasi: {len(result.get('recommendations', []))}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ [ERROR] {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Maaf, sistem sedang bermasalah.",
            "cards": [],
            "llm_response": "Maaf, terjadi kesalahan dalam memproses permintaan Anda.",
            "mode": "error"
        })

@app.route('/api/test', methods=['GET'])
def test_api():
    """Endpoint untuk testing API"""
    test_queries = [
        "Halo",
        "Siapa kamu?",
        "dimana pantai kaisarea",
        "fasilitas pantai kaisarea",
        "wisata di teminabuan",
        "wisata eksklusif",
        "wisata mahal",
        "pantai murah",
        "rekomendasi wisata keluarga"
    ]
    
    results = []
    for query in test_queries:
        try:
            result = recommender.process_user_query(query, top_k=2)
            results.append({
                "query": query,
                "results": len(result.get('recommendations', [])),
                "success": result.get('success', False),
                "mode": result.get('mode', 'unknown'),
                "status": "success" if result.get('success') else "error"
            })
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "status": "error"
            })
    
    return jsonify({
        "status": "ok",
        "total_queries": len(test_queries),
        "results": results
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Wisata Recommender API",
        "version": "1.0.0",
        "wisata_count": len(recommender.df) if hasattr(recommender, 'df') else 0,
        "dataset_count": len(recommender.raw_dataset) if hasattr(recommender, 'raw_dataset') else 0
    })

# =============================================
# STATIC FILE ROUTES
# =============================================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return app.send_static_file(filename)

# =============================================
# ERROR HANDLERS
# =============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route tidak ditemukan"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Terjadi kesalahan pada server"}), 500

# =============================================
# MAIN
# =============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 Flask App running on http://0.0.0.0:5000")
    print("=" * 60)
    print("\n📝 Contoh query yang bisa dicoba:")
    print("   • 'Halo' (sapaan)")
    print("   • 'Siapa kamu?' (tentang bot)")
    print("   • 'dimana pantai kaisarea' (lokasi spesifik)")
    print("   • 'fasilitas pantai kaisarea' (fasilitas spesifik)")
    print("   • 'wisata di teminabuan' (rekomendasi berdasarkan lokasi)")
    print("   • 'wisata eksklusif di raja ampat' (rekomendasi dengan filter)")
    print("   • 'wisata mahal untuk pasangan' (rekomendasi dengan filter harga)")
    print("   • 'pantai murah di sorong' (rekomendasi dengan filter lokasi & harga)")
    print("   • 'rekomendasi wisata keluarga dengan kolam renang' (rekomendasi lengkap)")
    print("\n✅ Buka browser dan kunjungi: http://localhost:5000")
    print("   Akan terbuka halaman home terlebih dahulu")
    print("   Klik tombol untuk masuk ke halaman chat/rekomendasi")
    app.run(host='0.0.0.0', port=5000, debug=True)