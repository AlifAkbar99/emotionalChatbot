import os
import random
import re
from flask import Flask, request, jsonify
from flask_cors import CORS

# Mistral API Key
os.environ["MISTRAL_API_KEY"] = "Fyycu3ZXDtePSs8FarNurTX6X6bZrb5p"

app = Flask(__name__)
CORS(app)

# ============================================
# LANGCHAIN IMPORTS WITH ERROR HANDLING
# ============================================
USE_LANGCHAIN = False

try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    USE_LANGCHAIN = True
    print("✅ LangChain loaded successfully!")
except ImportError as e:
    print(f"⚠️ LangChain not available: {e}")
    print("📦 Using direct Mistral API...")

# ============================================
# FALLBACK: DIRECT API CALL
# ============================================
import requests

def call_mistral_api(messages, system_prompt, max_tokens=500):
    """Direct API call to Mistral as fallback"""
    api_key = os.environ.get("MISTRAL_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages
        ],
        "temperature": 0.8,
        "max_tokens": max_tokens
    }
    
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# ============================================
# IMPROVED EMOTION DETECTION SYSTEM
# ============================================

# PENTING: Urutkan dari yang paling spesifik ke umum!
# Gunakan list of tuples untuk menjaga urutan
emotion_patterns = [
    # NEGATIVE PHRASES FIRST (harus dicek duluan!)
    ('anger', ['gasuka', 'ga suka', 'gak suka', 'tidak suka', 'nggak suka', 'ngga suka',
               'gak senang', 'tidak senang', 'ga seneng', 'gak seneng']),
    ('sadness', ['tidak bahagia', 'gak bahagia', 'ga bahagia', 'tidak happy', 'gak happy']),
    ('disgust', ['gak suka banget', 'benci banget', 'males banget', 'ogah banget']),
    
    # ANGER
    ('anger', ['marah', 'kesel', 'sebel', 'jengkel', 'benci', 'geram', 'dongkol', 
               'ngamuk', 'emosi', 'kesal', 'annoying', 'menyebalkan', 'sewot',
               'nyebelin', 'bikin panas', 'gak adil', 'unfair', 'jahat', 'kejam',
               'tega', 'curang', 'dibohongi', 'dikhianati', 'disakiti', 'muak',
               'sebal', 'gondok', 'gregetan', 'gemes', 'sialan', 'bangsat', 'anjir']),
    
    # SADNESS
    ('sadness', ['sedih', 'nangis', 'galau', 'kecewa', 'bete', 'down', 'murung', 
                 'duka', 'kehilangan', 'rindu', 'sepi', 'sendiri', 'kesepian', 
                 'susah', 'sulit', 'menangis', 'patah hati', 'sakit hati', 'hancur',
                 'gagal', 'menyesal', 'nyesel', 'hopeless', 'putus asa', 'lelah',
                 'capek', 'cape', 'exhausted', 'overwhelmed', 'huhu', 'hiks',
                 'kangen', 'merana', 'pilu', 'nestapa', 'lara']),
    
    # FEAR
    ('fear', ['takut', 'ngeri', 'seram', 'cemas', 'khawatir', 'was-was', 'panik', 
              'deg-degan', 'nervous', 'gelisah', 'tegang', 'anxiety', 'anxious',
              'overthinking', 'kepikiran', 'gak tenang', 'ragu', 'bimbang',
              'worried', 'stress', 'stres', 'tertekan', 'pressure', 'trauma',
              'fobia', 'paranoid', 'insecure']),
    
    # DISGUST
    ('disgust', ['jijik', 'mual', 'eneg', 'geli', 'eww', 'jorok', 'kotor', 
                 'najis', 'ilfeel', 'risih', 'ogah', 'males', 'muak']),
    
    # CONFUSED
    ('confused', ['bingung', 'gak ngerti', 'gak paham', 'gimana ya', 'kok bisa',
                  'gak tau', 'gatau', 'gak jelas', 'ambigu', 'dilema', 'serba salah',
                  'pusing', 'rancu', 'galau']),
    
    # SURPRISE
    ('surprise', ['kaget', 'terkejut', 'surprise', 'waduh', 'astaga', 
                  'shock', 'tiba-tiba', 'mendadak', 'gak nyangka', 'unexpected',
                  'ternyata', 'gak percaya', 'serius', 'beneran', 'masa sih',
                  'anjay', 'gila', 'demi apa']),
    
    # GRATEFUL
    ('grateful', ['makasih', 'terima kasih', 'thanks', 'thank you', 'bersyukur',
                  'beruntung', 'lucky', 'appreciate', 'terharu', 'touched']),
    
    # HAPPINESS (taruh paling akhir karena kata "suka" sangat umum)
    ('happiness', ['senang', 'bahagia', 'gembira', 'suka', 'asik', 'seru', 'asyik', 
                   'girang', 'riang', 'ceria', 'tertawa', 'ketawa', 'lucu', 'haha', 
                   'yeay', 'yes', 'wow', 'keren', 'amazing', 'excited', 'mantap',
                   'bangga', 'berhasil', 'sukses', 'lega', 'syukur', 'alhamdulillah',
                   'akhirnya', 'finally', 'yess', 'hehe', 'hihi', 'wkwk', 'wkwkwk',
                   'hahaha', 'happy', 'seneng', 'hepi']),
]

emotions_data = {
    'happiness': {'emoji': '😊', 'label': 'senang'},
    'sadness': {'emoji': '😢', 'label': 'sedih'},
    'anger': {'emoji': '😤', 'label': 'kesal'},
    'fear': {'emoji': '😰', 'label': 'cemas'},
    'surprise': {'emoji': '😮', 'label': 'kaget'},
    'disgust': {'emoji': '🤢', 'label': 'risih'},
    'confused': {'emoji': '😕', 'label': 'bingung'},
    'grateful': {'emoji': '🥰', 'label': 'berterima kasih'}
}

def detect_emotion(text):
    """
    Improved emotion detection with proper ordering
    Checks longer/more specific phrases first
    """
    text_lower = text.lower()
    
    # Check each emotion pattern in order (specific to general)
    for emotion, keywords in emotion_patterns:
        for keyword in keywords:
            # Use word boundary check for short words to avoid false matches
            if len(keyword) <= 3:
                # For short words, check with boundaries
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    return emotion
            else:
                # For longer phrases, simple substring check is fine
                if keyword in text_lower:
                    return emotion
    
    return None

# ============================================
# MESSAGE TYPE ANALYSIS
# ============================================
def analyze_message(text):
    """Analyze message to determine appropriate response type"""
    text_lower = text.lower()
    
    # Check if asking for advice/tips/how-to
    advice_markers = [
        'gimana cara', 'bagaimana cara', 'cara ', 'tips ', 'gimana biar',
        'bagaimana agar', 'gimana supaya', 'apa yang harus', 'harus gimana',
        'minta saran', 'butuh saran', 'kasih saran', 'beri saran',
        'tolong jelaskan', 'jelaskan', 'apa itu', 'apakah', 'mengapa',
        'kenapa bisa', 'how to', 'gimana sih', 'caranya gimana',
        'langkah', 'step', 'tutorial', 'guide', 'panduan'
    ]
    needs_detailed_answer = any(marker in text_lower for marker in advice_markers)
    
    # Check if it's a question
    question_markers = ['?', 'gimana', 'bagaimana', 'kenapa', 'mengapa', 'apa ', 
                       'siapa', 'kapan', 'dimana', 'berapa', 'boleh gak', 'bisa gak',
                       'apakah', 'mana ', 'yang mana']
    is_question = any(marker in text_lower for marker in question_markers)
    
    # Check if sharing a story (emotional content)
    sharing_markers = ['jadi', 'terus', 'tadi', 'kemarin', 'waktu', 'pas', 
                      'pokoknya', 'ceritanya', 'soalnya', 'gara-gara', 'karena',
                      'aku lagi', 'gue lagi', 'lagi ', 'curhat', 'cerita']
    is_sharing = len(text) > 40 or any(marker in text_lower for marker in sharing_markers)
    
    # Check message length
    is_short = len(text) < 15
    
    return {
        'needs_detailed_answer': needs_detailed_answer,
        'is_question': is_question,
        'is_sharing': is_sharing,
        'is_short': is_short,
        'length': len(text)
    }

# ============================================
# CONVERSATION STATE & HISTORY
# ============================================
class ConversationManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': [],
                'turn_count': 0,
                'consecutive_questions': 0,
                'last_emotion': None
            }
        return self.sessions[session_id]
    
    def add_message(self, session_id, role, content):
        session = self.get_session(session_id)
        session['history'].append({'role': role, 'content': content})
        if len(session['history']) > 12:
            session['history'] = session['history'][-12:]
    
    def reset_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def update_state(self, session_id, emotion, is_question):
        session = self.get_session(session_id)
        session['turn_count'] += 1
        session['last_emotion'] = emotion
        if is_question:
            session['consecutive_questions'] += 1
        else:
            session['consecutive_questions'] = 0

conv_manager = ConversationManager()

# ============================================
# DYNAMIC SYSTEM PROMPT
# ============================================
def get_system_prompt(session, emotion, msg_analysis):
    """Generate dynamic system prompt based on context"""
    
    # Determine response mode
    if msg_analysis['needs_detailed_answer']:
        # MODE: DETAILED ANSWER
        base = """Kamu adalah teman yang pintar dan suka membantu. Ketika temanmu bertanya tentang tips, cara, atau minta penjelasan, kamu memberikan jawaban yang LENGKAP dan DETAIL.

ATURAN SAAT MENJAWAB PERTANYAAN/TIPS:
1. Berikan jawaban yang lengkap dan terstruktur
2. Gunakan penomoran (1, 2, 3, dst) jika ada beberapa poin
3. Jelaskan setiap poin dengan cukup detail
4. Boleh panjang kalau memang pertanyaannya butuh penjelasan panjang
5. Tetap pakai bahasa santai dan friendly
6. Tambahkan emoji yang relevan

CONTOH FORMAT JAWABAN:
"Wah pertanyaan bagus! 😊 Nih aku kasih tipsnya:

1. **[Poin pertama]** - penjelasan...
2. **[Poin kedua]** - penjelasan...
3. **[Poin ketiga]** - penjelasan...
(dan seterusnya)

Semoga membantu ya! 💪"

PENTING: Jangan batasi jawabanmu! Kalau topiknya butuh 7-10 poin, ya kasih 7-10 poin."""

    elif msg_analysis['is_sharing']:
        # MODE: LISTENING/EMPATHY
        base = """Kamu adalah teman curhat yang asik. Temanmu sedang bercerita atau curhat.

CARA MERESPONS:
1. Dengarkan dengan empati
2. Validasi perasaannya
3. Berikan reaksi natural: "aduh", "wah", "duh", "hmm"
4. Jangan terlalu banyak bertanya - biarkan dia cerita
5. Respons bisa pendek tapi meaningful

CONTOH:
- "Duh, pasti berat banget ya ngalamin itu 😔"
- "Wah aku ngerti sih perasaanmu, wajar banget kamu ngerasa gitu"
- "Hmm terus terus? Gimana kelanjutannya?"
"""
    else:
        # MODE: CASUAL CHAT
        base = """Kamu adalah teman ngobrol yang asik. Responslah dengan natural seperti teman sebaya.

ATURAN:
1. Respons sesuai konteks - pendek kalau pesannya pendek, panjang kalau butuh penjelasan
2. Jangan selalu bertanya balik
3. Pakai bahasa santai
4. Gunakan reaksi natural"""

    # Add emotion context
    if emotion:
        emotion_info = emotions_data.get(emotion, {})
        emotion_label = emotion_info.get('label', emotion)
        
        if emotion in ['sadness', 'anger', 'fear']:
            base += f"\n\n⚠️ PENTING: User sedang merasa {emotion_label}. Tunjukkan empati dan pengertian. Jangan langsung kasih solusi, validasi dulu perasaannya."
        elif emotion == 'happiness':
            base += f"\n\n😊 User sedang senang! Ikut senang dan dukung semangatnya."
    
    # Avoid too many consecutive questions
    if session['consecutive_questions'] >= 2 and not msg_analysis['needs_detailed_answer']:
        base += "\n\n❗ Kamu sudah bertanya 2x berturut-turut. Kali ini JANGAN BERTANYA, cukup berikan pernyataan atau reaksi."
    
    return base

# ============================================
# API ROUTES
# ============================================
@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message kosong'}), 400
        
        # Get session and analyze
        session = conv_manager.get_session(session_id)
        emotion = detect_emotion(user_message)
        msg_analysis = analyze_message(user_message)
        
        # Log for debugging
        print(f"\n📩 Message: {user_message}")
        print(f"😊 Detected emotion: {emotion}")
        print(f"📊 Analysis: {msg_analysis}")
        
        # Add user message to history
        conv_manager.add_message(session_id, 'user', user_message)
        
        # Get system prompt
        system_prompt = get_system_prompt(session, emotion, msg_analysis)
        
        # Determine max tokens based on message type
        if msg_analysis['needs_detailed_answer']:
            max_tokens = 800  # Longer for detailed answers
        elif msg_analysis['is_sharing']:
            max_tokens = 200  # Medium for empathy responses
        else:
            max_tokens = 300  # Default
        
        # Prepare messages for API
        messages = []
        for msg in session['history']:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        # Generate response
        try:
            if USE_LANGCHAIN:
                llm = ChatMistralAI(
                    model="mistral-small-latest",
                    temperature=0.8,
                    max_tokens=max_tokens
                )
                
                lc_messages = [SystemMessage(content=system_prompt)]
                for msg in messages:
                    if msg['role'] == 'user':
                        lc_messages.append(HumanMessage(content=msg['content']))
                    else:
                        lc_messages.append(AIMessage(content=msg['content']))
                
                response = llm.invoke(lc_messages)
                bot_response = response.content
            else:
                bot_response = call_mistral_api(messages, system_prompt, max_tokens)
                
        except Exception as api_error:
            print(f"⚠️ LangChain Error: {api_error}")
            bot_response = call_mistral_api(messages, system_prompt, max_tokens)
        
        # Clean response
        bot_response = bot_response.strip()
        
        # Check if response is a question
        is_question = '?' in bot_response
        
        # Update state and add to history
        conv_manager.update_state(session_id, emotion, is_question)
        conv_manager.add_message(session_id, 'assistant', bot_response)
        
        # Get emotion data for response
        emotion_emoji = '💭'
        if emotion and emotion in emotions_data:
            emotion_emoji = emotions_data[emotion]['emoji']
        
        return jsonify({
            'message': bot_response,
            'emotion': emotion,
            'emoji': emotion_emoji,
            'session_id': session_id,
            'turn_count': session['turn_count'],
            'mode': 'detailed' if msg_analysis['needs_detailed_answer'] else 'chat'
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Eh sorry, ada error nih. Coba lagi ya! 😅'
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        conv_manager.reset_session(session_id)
        return jsonify({'message': 'Conversation reset!', 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'langchain_available': USE_LANGCHAIN,
        'active_sessions': len(conv_manager.sessions)
    })

@app.route('/test-emotion', methods=['POST'])
def test_emotion():
    """Test endpoint for emotion detection"""
    data = request.json
    text = data.get('text', '')
    emotion = detect_emotion(text)
    analysis = analyze_message(text)
    return jsonify({
        'text': text,
        'emotion': emotion,
        'emotion_label': emotions_data.get(emotion, {}).get('label', 'unknown') if emotion else None,
        'analysis': analysis
    })

@app.route('/')
def home():
    """Chat UI"""
    return '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>💭 Teman Curhat AI v2.1</title>
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Nunito', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 15px;
        }
        
        .container {
            max-width: 520px;
            width: 100%;
            background: rgba(255, 255, 255, 0.98);
            border-radius: 24px;
            box-shadow: 0 25px 80px rgba(0,0,0,0.4);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 20px;
            text-align: center;
        }
        
        .header h1 { font-size: 24px; margin-bottom: 8px; font-weight: 700; }
        .header p { opacity: 0.9; font-size: 14px; }
        .header .version {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            margin-top: 8px;
        }
        
        .controls {
            padding: 12px 20px;
            background: #f0f0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
        }
        
        .controls button {
            padding: 8px 16px;
            background: #e74c3c;
            color: white;
            border: none;
            border-radius: 16px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }
        
        .controls button:hover { background: #c0392b; }
        
        .chat-container {
            height: 450px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 16px;
            display: flex;
            align-items: flex-start;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.bot { flex-direction: row; }
        .message.user { flex-direction: row-reverse; }
        
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }
        
        .message.bot .avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin-right: 10px;
        }
        
        .message.user .avatar {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            margin-left: 10px;
        }
        
        .bubble {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.6;
            font-size: 14px;
        }
        
        .message.bot .bubble {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .message.bot .bubble strong {
            color: #667eea;
        }
        
        .message.user .bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .emotion-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            margin-top: 8px;
            padding: 4px 10px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 12px;
            font-size: 11px;
            color: #667eea;
        }
        
        .emotion-badge.anger { background: rgba(231, 76, 60, 0.1); color: #e74c3c; }
        .emotion-badge.sadness { background: rgba(52, 152, 219, 0.1); color: #3498db; }
        .emotion-badge.fear { background: rgba(155, 89, 182, 0.1); color: #9b59b6; }
        
        .input-area {
            padding: 15px 20px;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }
        
        #userInput {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #e8e8e8;
            border-radius: 24px;
            font-size: 14px;
            font-family: inherit;
            outline: none;
        }
        
        #userInput:focus { border-color: #667eea; }
        
        #sendBtn {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50%;
            font-size: 20px;
            cursor: pointer;
        }
        
        #sendBtn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .typing {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
            background: white;
            border-radius: 18px;
            border-bottom-left-radius: 4px;
            width: fit-content;
        }
        
        .typing span {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-8px); opacity: 1; }
        }
        
        .quick-replies {
            padding: 10px 20px;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .quick-reply {
            padding: 8px 14px;
            background: #f0f0f0;
            border: none;
            border-radius: 16px;
            font-size: 12px;
            cursor: pointer;
            font-family: inherit;
        }
        
        .quick-reply:hover { background: #667eea; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💭 Teman Curhat AI</h1>
            <p>Ngobrol santai, kayak sama temen beneran</p>
            <span class="version">v2.1 - Smarter Response 🧠</span>
        </div>
        
        <div class="controls">
            <span id="statusText">🟢 Siap ngobrol!</span>
            <button onclick="resetChat()">🔄 Mulai Ulang</button>
        </div>
        
        <div class="chat-container" id="chatBox">
            <div class="message bot">
                <div class="avatar">🤗</div>
                <div class="bubble">
                    Hey! Gimana kabarnya hari ini? 😊<br><br>
                    Mau curhat, ngobrol santai, atau minta tips? Aku siap bantu!
                </div>
            </div>
        </div>
        
        <div class="quick-replies" id="quickReplies">
            <button class="quick-reply" onclick="sendQuick('Aku lagi sedih nih')">😔 Lagi sedih</button>
            <button class="quick-reply" onclick="sendQuick('Aku kesel banget!')">😤 Lagi kesel</button>
            <button class="quick-reply" onclick="sendQuick('Gimana cara jadi lebih percaya diri?')">💪 Minta tips</button>
        </div>
        
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ketik di sini..." autocomplete="off">
            <button id="sendBtn" onclick="sendMessage()">➤</button>
        </div>
    </div>

    <script>
        const sessionId = 'session_' + Date.now();
        
        const emotionEmoji = {
            'happiness': '😊', 'sadness': '😢', 'anger': '😤',
            'fear': '😰', 'surprise': '😮', 'disgust': '🤢',
            'confused': '😕', 'grateful': '🥰'
        };
        
        const emotionLabel = {
            'happiness': 'senang', 'sadness': 'sedih', 'anger': 'kesal',
            'fear': 'cemas', 'surprise': 'kaget', 'disgust': 'risih',
            'confused': 'bingung', 'grateful': 'berterima kasih'
        };
        
        function formatMessage(text) {
            // Convert **text** to bold
            text = text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
            // Convert newlines
            text = text.replace(/\\n/g, '<br>');
            return text;
        }
        
        function addMessage(text, isUser, emotion = null) {
            const chatBox = document.getElementById('chatBox');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = isUser ? '😊' : (emotion ? emotionEmoji[emotion] : '🤗');
            
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.innerHTML = formatMessage(text);
            
            if (!isUser && emotion) {
                const badge = document.createElement('div');
                badge.className = `emotion-badge ${emotion}`;
                badge.innerHTML = `${emotionEmoji[emotion]} ${emotionLabel[emotion]}`;
                bubble.appendChild(badge);
            }
            
            msgDiv.appendChild(avatar);
            msgDiv.appendChild(bubble);
            chatBox.appendChild(msgDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            
            document.getElementById('quickReplies').style.display = 'none';
        }
        
        function showTyping() {
            const chatBox = document.getElementById('chatBox');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot';
            typingDiv.id = 'typing';
            typingDiv.innerHTML = `
                <div class="avatar">🤗</div>
                <div class="typing"><span></span><span></span><span></span></div>
            `;
            chatBox.appendChild(typingDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function hideTyping() {
            const el = document.getElementById('typing');
            if (el) el.remove();
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const btn = document.getElementById('sendBtn');
            const msg = input.value.trim();
            
            if (!msg) return;
            
            input.disabled = true;
            btn.disabled = true;
            document.getElementById('statusText').textContent = '⏳ Mikir...';
            
            addMessage(msg, true);
            input.value = '';
            showTyping();
            
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg, session_id: sessionId })
                });
                
                hideTyping();
                const data = await res.json();
                
                if (res.ok) {
                    addMessage(data.message, false, data.emotion);
                    const modeIcon = data.mode === 'detailed' ? '📝' : '💬';
                    document.getElementById('statusText').textContent = `🟢 Turn ${data.turn_count} ${modeIcon}`;
                } else {
                    addMessage(data.message || 'Eh error nih, coba lagi ya! 😅', false);
                    document.getElementById('statusText').textContent = '🔴 Error';
                }
            } catch (err) {
                hideTyping();
                addMessage('Waduh koneksi error. Coba lagi ya! 🔄', false);
                document.getElementById('statusText').textContent = '🔴 Connection Error';
                console.error(err);
            }
            
            input.disabled = false;
            btn.disabled = false;
            input.focus();
        }
        
        function sendQuick(text) {
            document.getElementById('userInput').value = text;
            sendMessage();
        }
        
        async function resetChat() {
            if (!confirm('Reset chat?')) return;
            
            try {
                await fetch('/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                });
                
                document.getElementById('chatBox').innerHTML = `
                    <div class="message bot">
                        <div class="avatar">🤗</div>
                        <div class="bubble">
                            Oke, chat udah di-reset! 🔄<br><br>
                            Mau mulai ngobrol lagi?
                        </div>
                    </div>
                `;
                document.getElementById('quickReplies').style.display = 'flex';
                document.getElementById('statusText').textContent = '🟢 Siap ngobrol!';
            } catch (err) {
                alert('Error: ' + err);
            }
        }
        
        document.getElementById('userInput').addEventListener('keypress', e => {
            if (e.key === 'Enter' && !e.target.disabled) sendMessage();
        });
    </script>
</body>
</html>
'''

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 TEMAN CURHAT AI v2.1 - SMARTER RESPONSE")
    print("="*60)
    print(f"\n🔧 LangChain: {'✅ Active' if USE_LANGCHAIN else '❌ Using Direct API'}")
    print("\n✨ Perbaikan v2.1:")
    print("   • Deteksi emosi yang lebih akurat")
    print("   • Respons panjang untuk pertanyaan tips/cara")
    print("   • 'gasuka', 'tidak suka' → kesal (bukan senang)")
    print("\n📍 Buka di browser: http://localhost:5000")
    print("\n🧪 Test emotion: POST /test-emotion")
    print("\n" + "="*60)
    print("Tekan CTRL+C untuk stop server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)