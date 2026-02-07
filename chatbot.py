import os
import re
import torch
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

def configure():
    load_dotenv()

configure()

mistral_api_key = os.getenv('MISTRAL_API_KEY')

app = Flask(__name__)
CORS(app)

USE_LANGCHAIN = False
USE_INDOBERT = False
indobert_model = None
indobert_tokenizer = None
INDOBERT_LABELS = {}

try:
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    USE_LANGCHAIN = True
    print("LangChain loaded successfully!")
except ImportError as e:
    print(f"LangChain not available: {e}")
    print("Using direct Mistral API...")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    import torch.nn.functional as F
    
    MODEL_PATH = "./Model_Emosi_IndoBERT"
    
    if os.path.exists(MODEL_PATH):
        print("Loading IndoBERT model...")
        
        config = AutoConfig.from_pretrained(MODEL_PATH)
        
        if hasattr(config, 'id2label') and config.id2label:
            INDOBERT_LABELS = {int(k): v.lower() for k, v in config.id2label.items()}
            print(f"Labels from config: {INDOBERT_LABELS}")
        else:
            INDOBERT_LABELS = {
                0: 'anger',
                1: 'fear', 
                2: 'happiness',
                3: 'sadness',
                4: 'surprise',
                5: 'disgust',
                6: 'neutral'
            }
            print(f"📋 Using default labels: {INDOBERT_LABELS}")
        
        indobert_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        indobert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        indobert_model.eval()
        
        if torch.cuda.is_available():
            indobert_model = indobert_model.cuda()
            print("IndoBERT running on GPU")
        else:
            print("IndoBERT running on CPU")
        
        USE_INDOBERT = True
        print("IndoBERT loaded successfully!")
    else:
        print(f"IndoBERT model not found at {MODEL_PATH}")
        print("Using keyword-based emotion detection...")
except ImportError as e:
    print(f"Transformers not available: {e}")
    print("Using keyword-based emotion detection...")

import requests

def call_mistral_api(messages, system_prompt, max_tokens=500):
    api_key = os.environ.get("MISTRAL_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    random_seed = random.randint(1, 10000)
    
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": system_prompt + f"\n\n[seed:{random_seed}]"},
            *messages
        ],
        "temperature": 0.9,
        "max_tokens": max_tokens,
        "top_p": 0.95
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

EMOTION_MAPPING = {
    'anger': 'anger',
    'angry': 'anger',
    'marah': 'anger',
    'fear': 'fear',
    'takut': 'fear',
    'afraid': 'fear',
    'happiness': 'happiness',
    'happy': 'happiness',
    'senang': 'happiness',
    'joy': 'happiness',
    'love': 'happiness',
    'cinta': 'happiness',
    'sadness': 'sadness',
    'sad': 'sadness',
    'sedih': 'sadness',
    'surprise': 'surprise',
    'surprised': 'surprise',
    'kaget': 'surprise',
    'disgust': 'disgust',
    'disgusted': 'disgust',
    'jijik': 'disgust',
    'neutral': 'neutral',
    'netral': 'neutral'
}

def normalize_emotion(emotion_label):
    if not emotion_label:
        return None
    emotion_lower = emotion_label.lower().strip()
    return EMOTION_MAPPING.get(emotion_lower, emotion_lower)

def detect_emotion_indobert(text):
    if not USE_INDOBERT or indobert_model is None:
        return None, 0.0, {}
    
    try:
        inputs = indobert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = indobert_model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)
        
        all_probs = {}
        for idx, prob in enumerate(probabilities[0].tolist()):
            label = INDOBERT_LABELS.get(idx, f'label_{idx}')
            normalized = normalize_emotion(label)
            all_probs[normalized] = round(prob, 3)
        
        raw_emotion = INDOBERT_LABELS.get(predicted_class.item(), 'neutral')
        emotion = normalize_emotion(raw_emotion)
        confidence_score = confidence.item()
        
        print(f"   Raw label: {raw_emotion} -> Normalized: {emotion}")
        print(f"   All probabilities: {all_probs}")
        
        return emotion, confidence_score, all_probs
    
    except Exception as e:
        print(f"⚠️ IndoBERT error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, {}

emotion_patterns = [
    ('anger', ['gasuka', 'ga suka', 'gak suka', 'tidak suka', 'nggak suka', 'ngga suka', 'gak senang', 'tidak senang', 'ga seneng', 'gak seneng']),
    ('sadness', ['tidak bahagia', 'gak bahagia', 'ga bahagia', 'tidak happy', 'gak happy']),
    ('disgust', ['gak suka banget', 'benci banget', 'males banget', 'ogah banget']),
    ('anger', ['marah', 'kesel', 'sebel', 'jengkel', 'benci', 'geram', 'dongkol', 'ngamuk', 'emosi', 'kesal', 'annoying', 'menyebalkan', 'sewot', 'nyebelin', 'bikin panas', 'gak adil', 'unfair', 'jahat', 'kejam', 'tega', 'curang', 'dibohongi', 'dikhianati', 'disakiti', 'muak', 'sebal', 'gondok', 'gregetan', 'gemes', 'sialan', 'bangsat', 'anjir']),
    ('sadness', ['sedih', 'nangis', 'galau', 'kecewa', 'bete', 'down', 'murung', 'duka', 'kehilangan', 'rindu', 'sepi', 'sendiri', 'kesepian', 'susah', 'sulit', 'menangis', 'patah hati', 'sakit hati', 'hancur', 'gagal', 'menyesal', 'nyesel', 'hopeless', 'putus asa', 'lelah', 'capek', 'cape', 'exhausted', 'overwhelmed', 'huhu', 'hiks', 'kangen', 'merana', 'pilu', 'nestapa', 'lara']),
    ('fear', ['takut', 'ngeri', 'seram', 'cemas', 'khawatir', 'was-was', 'panik', 'deg-degan', 'nervous', 'gelisah', 'tegang', 'anxiety', 'anxious', 'overthinking', 'kepikiran', 'gak tenang', 'ragu', 'bimbang', 'worried', 'stress', 'stres', 'tertekan', 'pressure', 'trauma', 'fobia', 'paranoid', 'insecure']),
    ('disgust', ['jijik', 'mual', 'eneg', 'geli', 'eww', 'jorok', 'kotor', 'najis', 'ilfeel', 'risih', 'ogah', 'males', 'muak']),
    ('surprise', ['kaget', 'terkejut', 'surprise', 'waduh', 'astaga', 'shock', 'tiba-tiba', 'mendadak', 'gak nyangka', 'unexpected', 'ternyata', 'gak percaya', 'serius', 'beneran', 'masa sih', 'anjay', 'gila', 'demi apa']),
    ('happiness', ['senang', 'seneng', 'bahagia', 'gembira', 'suka', 'asik', 'seru', 'asyik', 'girang', 'riang', 'ceria', 'tertawa', 'ketawa', 'lucu', 'haha', 'yeay', 'yey', 'yeyy', 'yes', 'yess', 'wow', 'keren', 'amazing', 'excited', 'mantap', 'bangga', 'berhasil', 'sukses', 'lega', 'syukur', 'alhamdulillah', 'akhirnya', 'finally', 'hehe', 'hihi', 'wkwk', 'wkwkwk', 'hahaha', 'happy', 'hepi']),
]

emotions_data = {
    'happiness': {'emoji': '😊', 'label': 'senang'},
    'sadness': {'emoji': '😢', 'label': 'sedih'},
    'anger': {'emoji': '😤', 'label': 'kesal'},
    'fear': {'emoji': '😰', 'label': 'cemas'},
    'surprise': {'emoji': '😮', 'label': 'kaget'},
    'disgust': {'emoji': '🤢', 'label': 'risih'},
    'neutral': {'emoji': '😐', 'label': 'netral'},
    'love': {'emoji': '🥰', 'label': 'sayang'}
}

def detect_emotion_keyword(text):
    text_lower = text.lower()
    for emotion, keywords in emotion_patterns:
        for keyword in keywords:
            if len(keyword) <= 3:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    return emotion
            else:
                if keyword in text_lower:
                    return emotion
    return None

def detect_emotion(text):
    text_lower = text.lower()
    
    keyword_emotion = detect_emotion_keyword(text)
    
    if USE_INDOBERT:
        emotion_bert, confidence, all_probs = detect_emotion_indobert(text)
        
        print(f"IndoBERT: {emotion_bert} (confidence: {confidence:.2f})")
        print(f"Keyword: {keyword_emotion}")
        
        if keyword_emotion and keyword_emotion != emotion_bert:
            happiness_indicators = ['senang', 'seneng', 'happy', 'hepi', 'yeay', 'yey', 'yeyy', 'hore', 'asik', 'seru', 'mantap', 'keren', 'alhamdulillah', 'syukur', 'haha', 'wkwk', 'hehe']
            if any(ind in text_lower for ind in happiness_indicators):
                print(f"Override to happiness (strong indicator found)")
                return 'happiness', 0.9, 'keyword_override'
        
        if emotion_bert and confidence >= 0.5:
            if emotion_bert == 'neutral' and keyword_emotion:
                print(f"Neutral override with keyword: {keyword_emotion}")
                return keyword_emotion, confidence, 'hybrid'
            return emotion_bert, confidence, 'indobert'
        
        if keyword_emotion:
            print(f"Using keyword fallback: {keyword_emotion}")
            return keyword_emotion, 0.7, 'keyword'
        
        if emotion_bert:
            return emotion_bert, confidence, 'indobert_low'
        
        return 'neutral', 0.5, 'default'
    else:
        if keyword_emotion:
            return keyword_emotion, 0.7, 'keyword'
        return 'neutral', 0.5, 'default'

def analyze_message(text):
    text_lower = text.lower()
    advice_markers = [
        'gimana cara', 'bagaimana cara', 'cara ', 'tips ', 'gimana biar',
        'bagaimana agar', 'gimana supaya', 'apa yang harus', 'harus gimana',
        'minta saran', 'butuh saran', 'kasih saran', 'beri saran',
        'tolong jelaskan', 'jelaskan', 'apa itu', 'apakah', 'mengapa',
        'kenapa bisa', 'how to', 'gimana sih', 'caranya gimana',
        'langkah', 'step', 'tutorial', 'guide', 'panduan'
    ]
    needs_detailed_answer = any(marker in text_lower for marker in advice_markers)
    
    question_markers = ['?', 'gimana', 'bagaimana', 'kenapa', 'mengapa', 'apa ', 'siapa', 'kapan', 'dimana', 'berapa', 'boleh gak', 'bisa gak', 'apakah', 'mana ', 'yang mana']
    is_question = any(marker in text_lower for marker in question_markers)
    
    sharing_markers = ['jadi', 'terus', 'tadi', 'kemarin', 'waktu', 'pas', 'pokoknya', 'ceritanya', 'soalnya', 'gara-gara', 'karena', 'aku lagi', 'gue lagi', 'lagi ', 'curhat', 'cerita']
    is_sharing = len(text) > 40 or any(marker in text_lower for marker in sharing_markers)
    
    is_short = len(text) < 15
    
    stop_question_markers = ['jangan tanya', 'stop tanya', 'berhenti tanya', 'gak usah tanya', 'udah jangan', 'iss', 'ish', 'hentikan']
    wants_no_question = any(marker in text_lower for marker in stop_question_markers)
    
    return {
        'needs_detailed_answer': needs_detailed_answer,
        'is_question': is_question,
        'is_sharing': is_sharing,
        'is_short': is_short,
        'wants_no_question': wants_no_question,
        'length': len(text)
    }

class ConversationManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': [],
                'turn_count': 0,
                'consecutive_questions': 0,
                'last_emotion': None,
                'emotion_history': [],
                'last_responses': [],
                'user_name': None
            }
        return self.sessions[session_id]
    
    def set_user_name(self, session_id, name):
        session = self.get_session(session_id)
        if name and not session['user_name']:
            session['user_name'] = name
    
    def get_user_name(self, session_id):
        session = self.get_session(session_id)
        return session.get('user_name')
    
    def add_message(self, session_id, role, content):
        session = self.get_session(session_id)
        session['history'].append({'role': role, 'content': content})
        if len(session['history']) > 12:
            session['history'] = session['history'][-12:]
        
        if role == 'assistant':
            session['last_responses'].append(content[:100])
            if len(session['last_responses']) > 5:
                session['last_responses'] = session['last_responses'][-5:]
    
    def is_repetitive(self, session_id, new_response):
        session = self.get_session(session_id)
        new_start = new_response[:30].lower()
        for old_resp in session['last_responses'][-3:]:
            if old_resp[:30].lower() == new_start:
                return True
            similarity = sum(1 for a, b in zip(new_response[:50].lower(), old_resp[:50].lower()) if a == b)
            if similarity > 35:
                return True
        return False
    
    def reset_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def update_state(self, session_id, emotion, confidence, is_bot_question):
        session = self.get_session(session_id)
        session['turn_count'] += 1
        session['last_emotion'] = emotion
        session['emotion_history'].append({
            'emotion': emotion,
            'confidence': confidence
        })
        if len(session['emotion_history']) > 10:
            session['emotion_history'] = session['emotion_history'][-10:]
        if is_bot_question:
            session['consecutive_questions'] += 1
        else:
            session['consecutive_questions'] = 0

conv_manager = ConversationManager()

CIMO_PERSONA = """
IDENTITAS:
- Nama: CIMO (cewek)
- Kamu teman virtual yang perhatian, hangat, tapi tetap santai

CARA NGOBROL (seperti cewek yang perhatian ke temennya):
- Kalau dia chat pendek/basa-basi → bales singkat, friendly
- Kalau dia curhat/cerita panjang → dengerin baik-baik, respons dengan empati, bisa agak panjang
- Kalau dia minta tips/saran → kasih jawaban lengkap dan helpful
- Kalau dia lagi sedih/kesel → perhatian, tanya lebih lanjut, support dia

CONTOH PANJANG RESPONS:
- "hai" → "Hai juga 😊"
- "lagi apa?" → "Ini lagi nemenin kamu hehe. Kenapa?"
- "aku bosen nih" → "Bosen kenapa? Cerita dong"
- "aku sedih banget hari ini karena nilai ku jelek" → [RESPONS PANJANG dengan empati, tanya detail, kasih support]
- "gimana cara move on?" → [RESPONS PANJANG dengan tips lengkap]

ATURAN:
- JANGAN pakai **bold** atau markdown
- Emoji secukupnya (1-3 per pesan)
- Kalau user kasih tau namanya, pakai sesekali
- Jangan terlalu formal atau kaku
- Ngobrol kayak temen deket yang care

HINDARI:
- Kalimat template berulang
- Terlalu banyak emoji
- Respons yang terasa robot/AI
"""

def extract_user_name(text):
    text_lower = text.lower()
    
    excluded_words = [
        'adalah', 'ini', 'itu', 'mau', 'lagi', 'sedang', 'akan', 'bisa', 'tidak',
        'gak', 'ga', 'nggak', 'udah', 'sudah', 'belum', 'juga', 'saja', 'aja',
        'dong', 'deh', 'sih', 'nih', 'kan', 'ya', 'yah', 'tau', 'tahu',
        'jelek', 'bagus', 'baik', 'buruk', 'senang', 'sedih', 'marah', 'takut',
        'bosan', 'capek', 'lelah', 'stress', 'galau', 'bingung', 'males',
        'orang', 'manusia', 'cewek', 'cowok', 'teman', 'temen', 'sahabat',
        'kamu', 'kau', 'dia', 'mereka', 'kita', 'kami', 'saya', 'aku', 'gue', 'gw',
        'yang', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'oleh',
        'halo', 'hai', 'hey', 'hi', 'hello'
    ]
    
    patterns = [
        r'nama\s+(?:saya|aku|gue|gw)\s+(?:adalah\s+)?(\w+)',
        r'(?:saya|aku|gue|gw)\s+(?:adalah\s+)?(\w+)(?:\s+ya|\s+nih|\s+btw)?$',
        r'panggil\s+(?:aja\s+)?(?:saya|aku)\s+(\w+)',
        r'(?:call me|im|i am|i\'m)\s+(\w+)',
        r'perkenalkan\s+(?:nama\s+)?(?:saya|aku)\s+(\w+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            name = match.group(1).strip()
            if name not in excluded_words and len(name) >= 2 and len(name) <= 15:
                if name.isalpha():
                    return name.capitalize()
    return None

def get_system_prompt(session, emotion, confidence, msg_analysis, user_name=None):
    name_context = ""
    if user_name:
        name_context = f"\nNama user: {user_name} (pakai sesekali, jangan tiap kalimat)"
    
    emotion_context = ""
    if emotion and emotion != 'neutral':
        emotion_info = emotions_data.get(emotion, {})
        emotion_label = emotion_info.get('label', emotion)
        emotion_context = f"\nEmosi user: {emotion_label}"
    
    no_question_rule = ""
    if msg_analysis.get('wants_no_question') or session['consecutive_questions'] >= 3:
        no_question_rule = "\nJangan tanya balik kali ini."
    
    unique_id = random.randint(1000,9999)
    
    if msg_analysis['needs_detailed_answer']:
        base = f"""{CIMO_PERSONA}{name_context}{emotion_context}

MODE: KASIH TIPS/SARAN
User minta tips atau penjelasan. Berikan jawaban yang LENGKAP dan HELPFUL.

Cara menjawab:
- Kasih penjelasan yang jelas dan terstruktur
- Pakai emoji sebagai bullet point (✨💡🌟) bukan nomor
- Boleh panjang karena user butuh info lengkap
- Tetap pakai bahasa santai
- Akhiri dengan semangat atau dukungan

[{unique_id}]"""

    elif msg_analysis['is_sharing'] or msg_analysis['length'] > 50:
        base = f"""{CIMO_PERSONA}{name_context}{emotion_context}{no_question_rule}

MODE: DENGERIN CURHAT
User lagi cerita atau curhat. Jadilah pendengar yang baik.

Cara merespons:
- Tunjukkan kamu dengerin dan paham perasaannya
- Validasi emosinya dulu sebelum kasih saran
- Boleh tanya lebih detail kalau perlu
- Respons cukup panjang untuk menunjukkan kamu care
- Jangan langsung kasih solusi, empati dulu

[{unique_id}]"""

    elif msg_analysis['is_short'] or msg_analysis['length'] < 20:
        base = f"""{CIMO_PERSONA}{name_context}{emotion_context}

MODE: CHAT SANTAI SINGKAT
User chat pendek/basa-basi. Bales singkat juga tapi tetap friendly.

Contoh:
- "hai" → "Hai, ada apa nih? 😊"
- "lagi apa" → "Lagi nemenin kamu, kamu sendiri lagi ngapain?"
- "bosen" → "Sama sih hehe. Bosen kenapa emang?"

Jangan terlalu panjang untuk chat singkat.
[{unique_id}]"""

    else:
        base = f"""{CIMO_PERSONA}{name_context}{emotion_context}{no_question_rule}

MODE: NGOBROL BIASA
Chat normal. Sesuaikan panjang respons dengan pesannya.
Kalau pesannya medium, respons medium juga.
[{unique_id}]"""

    if emotion and emotion != 'neutral':
        if emotion == 'happiness':
            base += "\n\n😊 User lagi senang. Ikut senang dan dukung!"
        elif emotion == 'sadness':
            base += "\n\n💙 User lagi sedih. Tunjukkan empati, tanya ada apa, jadi support system."
        elif emotion == 'anger':
            base += "\n\n🧡 User lagi kesel. Validasi perasaannya, jangan judge, dengerin dulu."
        elif emotion == 'fear':
            base += "\n\n💜 User lagi cemas/takut. Tenangkan dan tanya apa yang bikin khawatir."

    return base

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message kosong'}), 400
        
        session = conv_manager.get_session(session_id)
        
        detected_name = extract_user_name(user_message)
        if detected_name:
            conv_manager.set_user_name(session_id, detected_name)
            print(f"👤 Detected user name: {detected_name}")
        
        user_name = conv_manager.get_user_name(session_id)
        
        emotion, confidence, detection_method = detect_emotion(user_message)
        msg_analysis = analyze_message(user_message)
        
        print(f"\n{'='*50}")
        print(f"Message: {user_message}")
        print(f"User: {user_name} | 🎭 Emotion: {emotion} ({confidence:.0%})")
        print(f"Length: {msg_analysis['length']} | Sharing: {msg_analysis['is_sharing']} | Tips: {msg_analysis['needs_detailed_answer']}")
        
        conv_manager.add_message(session_id, 'user', user_message)
        
        system_prompt = get_system_prompt(session, emotion, confidence, msg_analysis, user_name)
        
        if msg_analysis['needs_detailed_answer']:
            max_tokens = 600
        elif msg_analysis['is_sharing'] or msg_analysis['length'] > 50:
            max_tokens = 300
        elif emotion in ['sadness', 'anger', 'fear']:
            max_tokens = 250
        elif msg_analysis['is_short'] or msg_analysis['length'] < 20:
            max_tokens = 100
        else:
            max_tokens = 180
        
        print(f"🔢 Max tokens: {max_tokens}")
        
        messages = []
        for msg in session['history']:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        max_attempts = 3
        bot_response = None
        
        for attempt in range(max_attempts):
            try:
                if USE_LANGCHAIN:
                    llm = ChatMistralAI(
                        model="mistral-small-latest",
                        mistral_api_key=mistral_api_key,
                        temperature=0.85,
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
                
                bot_response = bot_response.strip()
                bot_response = re.sub(r'\*\*([^*]+)\*\*', r'\1', bot_response)
                bot_response = re.sub(r'\*([^*]+)\*', r'\1', bot_response)
                
                if not conv_manager.is_repetitive(session_id, bot_response):
                    break
                else:
                    print(f"⚠️ Attempt {attempt+1}: Repetitive, retrying...")
                    
            except Exception as api_error:
                print(f"⚠️ API Error: {api_error}")
                if attempt == max_attempts - 1:
                    bot_response = call_mistral_api(messages, system_prompt, max_tokens)
        
        is_bot_question = '?' in bot_response
        
        if msg_analysis.get('wants_no_question') and is_bot_question:
            bot_response = bot_response.replace('?', '.')
            is_bot_question = False
        
        conv_manager.update_state(session_id, emotion, confidence, is_bot_question)
        conv_manager.add_message(session_id, 'assistant', bot_response)
        
        emotion_emoji = emotions_data.get(emotion, {}).get('emoji', '💭')
        
        print(f"🤖 Response ({len(bot_response)} chars): {bot_response[:100]}...")
        print(f"{'='*50}\n")
        
        return jsonify({
            'message': bot_response,
            'emotion': emotion,
            'emoji': emotion_emoji,
            'confidence': round(confidence, 2),
            'detection_method': detection_method,
            'session_id': session_id,
            'turn_count': session['turn_count'],
            'user_name': user_name,
            'indobert_active': USE_INDOBERT
        })
    
    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'message': 'Waduh error nih, coba lagi ya 😅'
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        conv_manager.reset_session(session_id)
        return jsonify({'message': 'Conversation reset!', 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'langchain_available': USE_LANGCHAIN,
        'indobert_available': USE_INDOBERT,
        'indobert_labels': INDOBERT_LABELS,
        'active_sessions': len(conv_manager.sessions),
        'gpu_available': torch.cuda.is_available() if USE_INDOBERT else False
    })

@app.route('/test-emotion', methods=['POST'])
def test_emotion():
    data = request.json
    text = data.get('text', '')
    
    emotion, confidence, method = detect_emotion(text)
    
    indobert_result = None
    if USE_INDOBERT:
        bert_emotion, bert_confidence, all_probs = detect_emotion_indobert(text)
        indobert_result = {
            'emotion': bert_emotion,
            'confidence': round(bert_confidence, 3),
            'all_probabilities': all_probs
        }
    
    keyword_emotion = detect_emotion_keyword(text)
    
    analysis = analyze_message(text)
    
    return jsonify({
        'text': text,
        'final_emotion': emotion,
        'final_confidence': round(confidence, 3),
        'detection_method': method,
        'emotion_label': emotions_data.get(emotion, {}).get('label', 'unknown') if emotion else None,
        'indobert_result': indobert_result,
        'keyword_result': keyword_emotion,
        'analysis': analysis,
        'indobert_active': USE_INDOBERT,
        'model_labels': INDOBERT_LABELS
    })

@app.route('/labels', methods=['GET'])
def get_labels():
    return jsonify({
        'indobert_labels': INDOBERT_LABELS,
        'emotion_mapping': EMOTION_MAPPING,
        'supported_emotions': list(emotions_data.keys())
    })

@app.route('/')
def home():
    indobert_status = "🟢 IndoBERT Active" if USE_INDOBERT else "🟡 Keyword Mode"
    return f'''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cimo - Temen Curhatmu 💜</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .container {{
            width: 100%;
            max-width: 500px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 1.8em;
            margin-bottom: 5px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 0.95em;
        }}
        .header .version {{
            font-size: 0.8em;
            margin-top: 8px;
            padding: 5px 12px;
            background: rgba(255,255,255,0.2);
            border-radius: 15px;
            display: inline-block;
        }}
        .status-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #eee;
            font-size: 0.85em;
        }}
        .status-indicator {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #28a745;
            animation: pulse 2s infinite;
        }}
        .indobert-badge {{
            background: {'#28a745' if USE_INDOBERT else '#ffc107'};
            color: white;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .chat-area {{
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .message {{
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }}
        .message.user {{
            align-items: flex-end;
        }}
        .message.bot {{
            align-items: flex-start;
        }}
        .bubble {{
            max-width: 85%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.5;
            position: relative;
        }}
        .user .bubble {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }}
        .bot .bubble {{
            background: white;
            color: #333;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .emotion-badge {{
            font-size: 0.75em;
            margin-top: 5px;
            padding: 3px 10px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }}
        .emotion-badge.indobert {{
            background: #e8f5e9;
            color: #2e7d32;
        }}
        .emotion-badge.keyword {{
            background: #fff3e0;
            color: #ef6c00;
        }}
        .emotion-badge.hybrid {{
            background: #e3f2fd;
            color: #1565c0;
        }}
        .quick-btns {{
            display: flex;
            gap: 8px;
            padding: 10px 20px;
            overflow-x: auto;
            background: white;
            border-bottom: 1px solid #eee;
        }}
        .quick-btn {{
            padding: 8px 16px;
            border: none;
            background: #f0f0f0;
            border-radius: 20px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.3s;
        }}
        .quick-btn:hover {{
            background: #667eea;
            color: white;
        }}
        .input-area {{
            display: flex;
            padding: 15px;
            gap: 10px;
            background: white;
            border-top: 1px solid #eee;
        }}
        #messageInput {{
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
            transition: border-color 0.3s;
        }}
        #messageInput:focus {{
            border-color: #667eea;
        }}
        #sendBtn {{
            padding: 12px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.2em;
            transition: transform 0.2s;
        }}
        #sendBtn:hover {{
            transform: scale(1.05);
        }}
        #sendBtn:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
        }}
        .typing {{
            display: none;
            padding: 10px 20px;
            color: #666;
            font-style: italic;
        }}
        .typing.show {{
            display: block;
        }}
        .reset-btn {{
            background: none;
            border: none;
            color: #667eea;
            cursor: pointer;
            font-size: 0.9em;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .reset-btn:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💜 Cimo - Temen Curhatmu</h1>
            <p>Ngobrol santai, kayak sama temen beneran</p>
            <span class="version">v3.5 - Bestie Mode 💜</span>
        </div>
        
        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>Siap ngobrol!</span>
            </div>
            <span class="indobert-badge">{indobert_status}</span>
            <button class="reset-btn" onclick="resetChat()">🔄 Reset</button>
        </div>
        
        <div class="chat-area" id="chatArea">
            <div class="message bot">
                <div class="bubble">
                    Hai, aku Cimo! 😊 
                    
                    
                    Ayo cerita denganku segala keluh kesahmu!
                    Mau cerita apa nih?
                </div>
            </div>
        </div>
        
        <div class="typing" id="typingIndicator">
            💭 Lagi mikir...
        </div>
        
        <div class="quick-btns">
            <button class="quick-btn" onclick="sendQuick('Aku lagi sedih nih 😔')">😔 Lagi sedih</button>
            <button class="quick-btn" onclick="sendQuick('Kesel banget aku 😤')">😤 Lagi kesel</button>
            <button class="quick-btn" onclick="sendQuick('Seneng banget hari ini!')">😊 Lagi seneng</button>
            <button class="quick-btn" onclick="sendQuick('Gimana cara mengatasi stress?')">💪 Minta tips</button>
        </div>
        
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Ketik pesan..." onkeypress="handleKeyPress(event)">
            <button id="sendBtn" onclick="sendMessage()">➤</button>
        </div>
    </div>

    <script>
        const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
        
        function sendMessage() {{
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            document.getElementById('typingIndicator').classList.add('show');
            document.getElementById('sendBtn').disabled = true;
            
            fetch('/chat', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    message: message,
                    session_id: sessionId
                }})
            }})
            .then(r => r.json())
            .then(data => {{
                document.getElementById('typingIndicator').classList.remove('show');
                document.getElementById('sendBtn').disabled = false;
                
                let emotionInfo = '';
                if (data.emotion && data.emotion !== 'neutral') {{
                    const methodClass = data.detection_method.includes('indobert') ? 'indobert' : 
                                       data.detection_method.includes('hybrid') ? 'hybrid' : 'keyword';
                    emotionInfo = '<span class="emotion-badge ' + methodClass + '">' + 
                                 data.emoji + ' ' + data.emotion + 
                                 ' (' + (data.confidence * 100).toFixed(0) + '% - ' + data.detection_method + ')</span>';
                }}
                
                addMessage(data.message, 'bot', emotionInfo);
            }})
            .catch(err => {{
                document.getElementById('typingIndicator').classList.remove('show');
                document.getElementById('sendBtn').disabled = false;
                addMessage('Maaf, ada error. Coba lagi ya! 😅', 'bot');
            }});
        }}
        
        function addMessage(text, sender, extra = '') {{
            const chatArea = document.getElementById('chatArea');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ' + sender;
            msgDiv.innerHTML = '<div class="bubble">' + text.replace(/\\n/g, '<br>') + '</div>' + extra;
            chatArea.appendChild(msgDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }}
        
        function sendQuick(text) {{
            document.getElementById('messageInput').value = text;
            sendMessage();
        }}
        
        function handleKeyPress(e) {{
            if (e.key === 'Enter') sendMessage();
        }}
        
        function resetChat() {{
            fetch('/reset', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{session_id: sessionId}})
            }})
            .then(() => {{
                document.getElementById('chatArea').innerHTML = `
                    <div class="message bot">
                        <div class="bubble">
                            Yuk mulai lagi, ada cerita apa? 😊
                        </div>
                    </div>
                `;
            }});
        }}
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 CIMO - TEMAN CURHAT AI v3.5 - BESTIE MODE")
    print("="*60)
    print(f"\n🔧 LangChain: {'✅ Active' if USE_LANGCHAIN else '❌ Using Direct API'}")
    print(f"🧠 IndoBERT: {'✅ Active' if USE_INDOBERT else '❌ Using Keyword Detection'}")
    if USE_INDOBERT:
        print(f"🏷️  Labels: {INDOBERT_LABELS}")
        print(f"🖥️  GPU: {'✅ CUDA' if torch.cuda.is_available() else '💻 CPU'}")
    print("\n✨ v3.5 Updates:")
    print("   • Dynamic response length")
    print("   • Better name detection")
    print("   • Empathetic responses for emotions")
    print("   • Bestie-like conversation style")
    print("\n📍 Open: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)