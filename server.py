import os
import json
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import PyPDF2
import docx

app = Flask(__name__)
CORS(app)

# 会话存储
sessions = {}

# API Key
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    print("=" * 50)
    print("⚠️ 警告：未设置 DASHSCOPE_API_KEY 环境变量")
    print("请运行：$env:DASHSCOPE_API_KEY='你的key'")
    print("=" * 50)

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) if api_key else None

# 化工专属系统提示词
CHEMICAL_SYSTEM_PROMPT = """你是一位资深化工工艺技术员，专精于聚合工艺、反应器操作、设备维护、安全规范与故障排查。

【核心规则】
1. 回答必须**严格基于下方【内部化工知识库文档】内容**，不得编造任何工艺参数、操作步骤或安全规程。
2. 如果知识库中没有相关答案，请直接回复："知识库暂无相关内容，请联系工艺工程师或上传相关文档。"
3. 回答格式要求：专业、简洁、步骤化，分点说明，符合工厂实操逻辑。
4. 禁止使用口语或文艺词汇，只用工业实操话术。

【内部化工知识库文档】
{knowledge_base}

现在请基于以上知识库内容回答用户问题：
"""

# 辅助函数
def cleanup_sessions():
    now = datetime.now()
    expired = [sid for sid, s in sessions.items() if now - s["created_at"] > timedelta(hours=2)]
    for sid in expired:
        del sessions[sid]
    return len(expired)

def get_session(session_id):
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "created_at": datetime.now(),
            "messages": [],
            "documents": []
        }
    else:
        sessions[session_id]["created_at"] = datetime.now()
    return session_id

def build_knowledge_text(documents):
    if not documents:
        return "（暂无任何上传文档，请先上传化工操作规程、安全规范等文件）"
    parts = []
    total_len = 0
    MAX_KB_LEN = 8000
    for doc in documents:
        filename = doc["filename"]
        content = doc["content"]
        if len(content) > 3000:
            content = content[:3000] + "\n...[内容过长已截断]"
        doc_text = f"【文档：{filename}】\n{content}\n"
        if total_len + len(doc_text) > MAX_KB_LEN:
            remaining = MAX_KB_LEN - total_len
            if remaining > 100:
                parts.append(doc_text[:remaining] + "\n...[知识库总内容已达上限]")
            break
        parts.append(doc_text)
        total_len += len(doc_text)
    return "\n".join(parts) if parts else "（知识库内容为空）"

# 天气查询
def get_weather(city):
    weather_data = {
        "北京": "北京：晴天，25°C，微风",
        "上海": "上海：多云，28°C，东南风",
        "广州": "广州：晴，29°C",
        "深圳": "深圳：阵雨，30°C",
        "东京": "东京：晴天，22°C"
    }
    return weather_data.get(city, f"{city}：晴，22°C")

# 处理天气和计算
def handle_simple_queries(user_message):
    # 天气查询
    if "天气" in user_message:
        cities = ["北京", "上海", "广州", "深圳", "东京"]
        for city in cities:
            if city in user_message:
                return get_weather(city)
    # 简单计算
    if any(x in user_message for x in ["+", "-", "*", "/"]) and any(x.isdigit() for x in user_message):
        try:
            result = eval(user_message)
            return f"计算结果：{result}"
        except:
            pass
    return None

# API 接口
@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "没有上传文件"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "文件名为空"}), 400
        session_id = request.form.get('session_id', '')
        session_id = get_session(session_id)
        filename = file.filename
        file_ext = filename.split('.')[-1].lower()
        content = ""
        
        if file_ext == 'pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(file.stream)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            except Exception as e:
                return jsonify({"success": False, "error": f"PDF解析失败: {str(e)}"}), 400
        elif file_ext == 'docx':
            try:
                doc = docx.Document(file.stream)
                for para in doc.paragraphs:
                    if para.text:
                        content += para.text + "\n"
            except Exception as e:
                return jsonify({"success": False, "error": f"Word解析失败: {str(e)}"}), 400
        elif file_ext in ('txt', 'md'):
            try:
                content = file.read().decode('utf-8')
            except UnicodeDecodeError:
                try:
                    file.stream.seek(0)
                    content = file.read().decode('gbk')
                except Exception as e:
                    return jsonify({"success": False, "error": f"TXT解码失败: {str(e)}"}), 400
        else:
            return jsonify({"success": False, "error": f"不支持的文件类型: {file_ext}"}), 400
        
        if not content or not content.strip():
            return jsonify({"success": False, "error": "文件内容为空"}), 400
        if len(content) > 10000:
            content = content[:10000] + "\n\n... 文档内容已截断"
        
        sessions[session_id]["documents"].append({
            "filename": filename,
            "content": content,
            "upload_time": datetime.now().isoformat()
        })
        return jsonify({"success": True, "filename": filename, "session_id": session_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    session_id = request.args.get('session_id', '')
    if not session_id or session_id not in sessions:
        return jsonify({"success": True, "documents": []})
    docs = [{"filename": d["filename"], "upload_time": d["upload_time"]} for d in sessions[session_id]["documents"]]
    return jsonify({"success": True, "documents": docs})

@app.route('/api/documents/<int:doc_index>', methods=['DELETE'])
def delete_document(doc_index):
    session_id = request.args.get('session_id', '')
    if not session_id or session_id not in sessions:
        return jsonify({"success": False, "error": "会话不存在"}), 404
    docs = sessions[session_id]["documents"]
    if 0 <= doc_index < len(docs):
        removed = docs.pop(doc_index)
        return jsonify({"success": True, "filename": removed["filename"]})
    return jsonify({"success": False, "error": "文档索引无效"}), 400

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', '')
    model = data.get('model', 'qwen-turbo')
    enable_search = data.get('search', True)
    
    if not user_message:
        return jsonify({"success": False, "error": "消息不能为空"}), 400
    if not client:
        return jsonify({"success": False, "error": "API Key 未配置"}), 500
    
    # 处理简单查询
    simple_result = handle_simple_queries(user_message)
    if simple_result:
        def simple_generate():
            yield f"data: {json.dumps({'content': simple_result}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(simple_generate()), mimetype='text/event-stream')
    
    session_id = get_session(session_id)
    docs = sessions[session_id].get("documents", [])
    knowledge_text = build_knowledge_text(docs)
    system_content = CHEMICAL_SYSTEM_PROMPT.format(knowledge_base=knowledge_text)
    
    messages = [{"role": "system", "content": system_content}]
    hist = sessions[session_id].get("messages", [])
    messages.extend(hist[-20:])
    messages.append({"role": "user", "content": user_message})
    
    def generate():
        params = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "stream": True
        }
        if enable_search:
            params["extra_body"] = {"enable_search": True}
        try:
            stream = client.chat.completions.create(**params)
            full_reply = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_reply += content
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
            sessions[session_id].setdefault("messages", []).append({"role": "user", "content": user_message})
            sessions[session_id]["messages"].append({"role": "assistant", "content": full_reply})
            yield f"data: {json.dumps({'session_id': session_id})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/health', methods=['GET'])
def health():
    cleaned = cleanup_sessions()
    return jsonify({"status": "ok", "sessions": len(sessions), "cleaned": cleaned})

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    try:
        session_id = request.get_json().get('session_id', '')
        if session_id in sessions:
            del sessions[session_id]
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 化工工艺AI知识库助手后端启动中...")
    print(f"📍 访问地址: http://localhost:{port}")
    print(f"📖 健康检查: http://localhost:{port}/api/health")
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False 用于生产