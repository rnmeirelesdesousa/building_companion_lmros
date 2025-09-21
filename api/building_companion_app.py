"""
ConstróiBot Chat Bot - Um assistente especializado em construção
Interface web interativa para assistência em construção usando MLflow e Azure OpenAI
"""

from flask import Flask, request, jsonify, render_template_string
import os
from datetime import datetime
import logging
# --- KEPT: Your original RAG Agent is the core of the backend ---
from src.agents.rag_agent import RAGAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- MERGED: Frontend HTML Template from your colleague ---
# This provides the new graphics, interactive elements, and Portuguese text.
CHAT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ConstróiBot Chat - Seu Especialista em Construção de Edifícios</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #3a7bd5 0%, #0f2027 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: #e3f2fd;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #1565c0 0%, #283e51 100%);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        .chat-header h1 { font-size: 1.8em; margin-bottom: 5px; }
        .chat-header p { opacity: 0.9; font-size: 0.9em; }
        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            background: #1976d2;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f0f7fa;
        }
        .message { margin-bottom: 15px; display: flex; align-items: flex-start; }
        .message.user { justify-content: flex-end; }
        .message.bot { justify-content: flex-start; }
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .message.user .message-content {
            background: #1976d2;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .message.bot .message-content {
            background: #e3f2fd;
            color: #0d47a1;
            border: 1px solid #90caf9;
            border-bottom-left-radius: 4px;
        }
        .message-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            font-size: 0.8em;
        }
        .message.user .message-avatar { background: #1976d2; order: 2; }
        .message.bot .message-avatar { background: #1565c0; order: 1; }
        .chat-input {
            padding: 20px;
            background: #e3f2fd;
            border-top: 1px solid #90caf9;
        }
        .input-group { display: flex; gap: 10px; }
        .input-group input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #90caf9;
            border-radius: 25px;
            font-size: 0.9em;
            outline: none;
            transition: border-color 0.3s;
            background: #f0f7fa;
            color: #0d47a1;
        }
        .input-group input:focus { border-color: #1976d2; }
        .input-group button {
            padding: 12px 20px;
            background: #1976d2;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s;
        }
        .input-group button:hover { background: #1565c0; }
        .input-group button:disabled { background: #b0bec5; cursor: not-allowed; }
        .welcome-message {
            text-align: center;
            color: #1565c0;
            padding: 40px 20px;
        }
        .welcome-message h3 { color: #1976d2; margin-bottom: 10px; }
        .quick-questions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
            justify-content: center;
        }
        .quick-question {
            background: #bbdefb;
            color: #1565c0;
            padding: 8px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            cursor: pointer;
            transition: background 0.3s;
            border: none;
        }
        .quick-question:hover { background: #1976d2; color: white; }
        .typing-indicator {
            display: none;
            padding: 10px 16px;
            background: #e3f2fd;
            border-radius: 18px;
            border: 1px solid #90caf9;
            width: fit-content;
            margin-bottom: 15px;
            color: #1565c0;
        }
        .typing-dots { display: inline-block; }
        .typing-dots span {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #1976d2;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        .error-message {
            color: #d32f2f;
            text-align: center;
            padding: 10px;
            background: #ffcdd2;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="status-indicator"></div>
            <h1>🏠 ConstróiBot Chat</h1>
            <p>O teu assistente especializado em construção</p>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <h3>Bem-vindo ao ConstróiBot!</h3>
                <p>Sou o seu especialista em construção. Pergunte-me sobre licenças, materiais, planejamento ou qualquer dúvida relacionada à construção da sua casa!</p>
                <div class="quick-questions">
                    <button class="quick-question" onclick="sendQuickQuestion(this)">Posso construir uma cave para habitação?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">Qual a altura mínima para o pé-direito de uma loja comercial?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">Qual a largura minima de uma escada num prédio de 4 apartamentos?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">Qual é o equipamento minimo obrigatorio para uma habitação?</button>
                </div>
            </div>
        </div>
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            ConstróiBot está a pensar...
        </div>
        <div class="chat-input">
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Pergunte sobre a construção da sua casa/edifício..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()" id="sendButton">Enviar</button>
            </div>
        </div>
    </div>
    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = isUser ? 'Você' : '🏠';
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            const formattedContent = content
                .replace(/\\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

            messageContent.innerHTML = formattedContent;
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            chatMessages.appendChild(messageDiv);
            scrollToBottom();
        }
        function showTyping() { typingIndicator.style.display = 'block'; scrollToBottom(); }
        function hideTyping() { typingIndicator.style.display = 'none'; }
        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = message;
            chatMessages.appendChild(errorDiv);
            scrollToBottom();
        }
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            const welcomeMessage = document.querySelector('.welcome-message');
            if (welcomeMessage) { welcomeMessage.remove(); }
            addMessage(message, true);
            messageInput.value = '';
            sendButton.disabled = true;
            showTyping();
            try {
                // ADAPTED: The new frontend sends requests to /chat
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: message })
                });
                if (!response.ok) { throw new Error(`Erro do servidor: ${response.status}`); }
                const data = await response.json();
                hideTyping();
                if (data.error) {
                    showError(data.error);
                } else {
                    // ADAPTED: The new frontend expects the answer in a "response" key
                    addMessage(data.response);
                }
            } catch (error) {
                hideTyping();
                showError('Desculpe, ocorreu um erro. Por favor, tente novamente.');
                console.error('Erro:', error);
            } finally {
                sendButton.disabled = false;
                messageInput.focus();
            }
        }
        function sendQuickQuestion(button) {
            messageInput.value = button.textContent;
            sendMessage();
        }
        function handleKeyPress(event) { if (event.key === 'Enter') { sendMessage(); } }
        messageInput.focus();
    </script>
</body>
</html>
"""

# --- Global Variables ---
rag_agent = None

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def chat_interface():
    """Serve the main chat interface."""
    return render_template_string(CHAT_HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ControiBot-api", # KEPT: Portuguese Naming
        "agent_status": "loaded" if rag_agent else "not_loaded"
    })

# --- MERGED AND ADAPTED: Main chat endpoint ---
# The route is now /chat and the response key is "response" to match the new frontend.
@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint principal do ConstróiBot"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Falta o campo 'query'"}), 400

        user_query = data['query']
        if not user_query.strip():
            return jsonify({"error": "A pergunta não pode estar vazia"}), 400

        logger.info(f"🏗️ ConstróiBot a processar: {user_query[:50]}...")

        # KEPT: Your original RAG agent logic is preserved.
        answer = rag_agent.ask(user_query)
        logger.info("✅ Resposta ConstróiBot gerada")

        return jsonify({
            "query": user_query,
            "response": answer, # Formerly "answer", changed to "response"
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Erro no endpoint ConstróiBot: {str(e)}")
        return jsonify({"error": "Estou com dificuldades. Tenta novamente."}), 500

# --- Application Initialization ---
def initialize_app():
    """Inicializa a aplicação ConstróiBot."""
    global rag_agent
    logger.info("🏗️ Inicializando ConstróiBot Chat Bot...")
    try:
        # KEPT: Your RAG agent is loaded on startup as before.
        rag_agent = RAGAgent()
        logger.info("✅ RAG Agent carregado com sucesso.")
    except Exception as e:
        logger.error(f"❌ Falha ao inicializar o RAG Agent: {e}")
    logger.info("✅ ConstróiBot inicializado com sucesso.")
    logger.info("🌐 Aceda à interface em: http://localhost:3000")


if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=3000, debug=True)