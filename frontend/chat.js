/**
 * ScriptBees RAG Chatbot Frontend - improved
 * - Prevents page reloads
 * - Robust DOM init
 * - Connection status updates
 * - Better error handling
 */

// ========== CONFIGURATION ==========
let API_URL = 'https://chat-bot-6sgm.onrender.com'; // <-- update to your deployed backend (https://your-backend.example.com)
let API_KEY = 'X2Cli1ZSPhHHAHlfZkOEPRWIqtd1TQD9ErH705-HMc4'; // <-- replace with production key or set via runtime

// ========== STATE & DOM refs (initialized in init) ==========
let chatMessages, chatForm, userInput, sendButton;
let statusDot, statusText;

// ========== MESSAGE HELPERS ==========
function addMessage(text, isUser = false, sources = null, isError = false, metadata = null) {
    if (!chatMessages) return;
    const msg = document.createElement('div');
    msg.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const content = document.createElement('div');
    content.className = 'message-content';
    if (isError) content.classList.add('error-message');

    const textElem = document.createElement('p');
    textElem.innerHTML = String(text).replace(/\n/g, '<br>');
    content.appendChild(textElem);

    if (sources && sources.length > 0) {
        const srcDiv = document.createElement('div');
        srcDiv.className = 'sources';
        const title = document.createElement('div');
        title.className = 'sources-title';
        title.textContent = '📚 Sources from ScriptBees.com:';
        srcDiv.appendChild(title);
        sources.forEach(url => {
            const link = document.createElement('a');
            link.href = url;
            link.textContent = (url || '').replace('https://scriptbees.com', '').replace('https://www.scriptbees.com', '') || '/';
            link.className = 'source-link';
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            srcDiv.appendChild(link);
        });
        content.appendChild(srcDiv);
    }

    if (metadata) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'metadata';
        metaDiv.style.fontSize = '0.75rem';
        metaDiv.style.color = '#64748b';
        metaDiv.style.marginTop = '8px';
        metaDiv.style.fontStyle = 'italic';
        let metaText = '';
        if (metadata.response_time_seconds) metaText += `⚡ ${metadata.response_time_seconds.toFixed(2)}s`;
        if (metadata.cached) metaText += (metaText ? ' • ' : '') + '💾 Cached';
        if (metaText) {
            metaDiv.textContent = metaText;
            content.appendChild(metaDiv);
        }
    }

    msg.appendChild(content);
    chatMessages.appendChild(msg);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

function showTyping() {
    if (!chatMessages) return;
    removeTyping();
    const typing = document.createElement('div');
    typing.className = 'message bot-message';
    typing.id = 'typing-indicator';
    const content = document.createElement('div');
    content.className = 'message-content typing-indicator';
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        content.appendChild(dot);
    }
    typing.appendChild(content);
    chatMessages.appendChild(typing);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTyping() {
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.remove();
}

// ========== STATUS UI ==========
function setStatus(connected) {
    try {
        if (!statusDot || !statusText) {
            statusDot = document.getElementById('status-dot');
            statusText = document.getElementById('status-text');
        }
        if (!statusDot || !statusText) return;
        if (connected) {
            statusDot.classList.add('online');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('online');
            statusText.textContent = 'Disconnected';
        }
    } catch (e) {
        // ignore
        console.warn('setStatus error', e);
    }
}

// ========== API CALLS ==========
async function sendQuery(question) {
    const payload = { question };
    try {
        const response = await fetch(`${API_URL.replace(/\/+$/, '')}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            // attempt to parse JSON error
            let err = {};
            try { err = await response.json(); } catch (e) { /* ignore */ }
            if (response.status === 403) throw new Error('Invalid API key. Please check your configuration.');
            if (response.status === 503) throw new Error('Server still loading models. Please wait...');
            if (response.status === 404) throw new Error('API endpoint not found. Check backend is running.');
            throw new Error(err.detail || err.error || `Server error (${response.status})`);
        }

        const json = await response.json();
        return json;
    } catch (err) {
        const m = String(err.message || err || 'Network error');
        if (m.includes('Failed to fetch') || m.includes('NetworkError') || m.includes('NetworkError')) {
            throw new Error(`Cannot connect to backend at ${API_URL}\nCheck backend, CORS, and network access.`);
        }
        throw err;
    }
}

async function checkHealth() {
    try {
        const res = await fetch(`${API_URL.replace(/\/+$/, '')}/health`);
        if (!res.ok) throw new Error(`Health check HTTP ${res.status}`);
        const data = await res.json();
        console.log('🐝 ScriptBees RAG Server Health:', data);

        // update status UI
        if (data && data.status === 'healthy' && data.retriever_loaded && data.generator_loaded) {
            setStatus(true);
        } else {
            setStatus(false);
        }

        // if models still loading, show message and poll
        if (data && (!data.retriever_loaded || !data.generator_loaded)) {
            addMessage('⚠️ Server starting up. AI models are loading... This may take a minute or two.', false);
            setTimeout(checkHealth, 5000);
        }

        return data;
    } catch (err) {
        console.warn('Health check failed', err);
        setStatus(false);
        // Only show the "cannot connect" message once to avoid spamming
        // (If you want repeated messages, remove the guard)
        addMessage(`❌ Cannot connect to backend at ${API_URL}\nPlease ensure backend is running and CORS allows this origin.`, false, null, true);
        return null;
    }
}

// ========== SUBMIT HANDLING ==========
async function handleSubmit(e) {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();
    const question = userInput.value.trim();
    if (!question) return;

    addMessage(question, true);
    userInput.value = '';
    userInput.disabled = true;
    sendButton.disabled = true;
    sendButton.textContent = 'Thinking...';
    showTyping();

    try {
        const response = await sendQuery(question);
        removeTyping();
        addMessage(
            response.answer || "No answer returned.",
            false,
            response.sources || [],
            false,
            { response_time_seconds: response.response_time_seconds || 0, cached: !!response.cached }
        );
    } catch (err) {
        removeTyping();
        addMessage(`⚠️ Error:\n${err.message}`, false, null, true);
        console.error('Query error:', err);
    } finally {
        userInput.disabled = false;
        sendButton.disabled = false;
        sendButton.textContent = 'Send';
        userInput.focus();
    }
}

// ========== KEYBOARD SHORTCUTS ==========
function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
    }
}

// ========== INITIALIZATION ==========
function init() {
    // DOM refs - do them here so script works regardless of load order
    chatMessages = document.getElementById('chat-messages');
    chatForm = document.getElementById('chat-form');
    userInput = document.getElementById('user-input');
    sendButton = document.getElementById('send-button');
    statusDot = document.getElementById('status-dot');
    statusText = document.getElementById('status-text');

    // Attach listeners
    if (chatForm) {
        // prevent default if user presses submit in a browser that still treats button as submit
        chatForm.addEventListener('submit', handleSubmit);
    }
    if (sendButton) {
        sendButton.addEventListener('click', handleSubmit);
    }
    if (userInput) {
        userInput.addEventListener('keydown', handleKeydown);
    }

    // Console logs for debugging
    console.log('🐝 ScriptBees RAG Chatbot Initialized');
    console.log(`Backend: ${API_URL}`);
    console.log(`API Key: ${API_KEY ? API_KEY.substring(0, 8) + '...' : '(empty)'}`);

    // Health check + welcome message
    checkHealth();

    setTimeout(() => {
        addMessage(
            "👋 Welcome to ScriptBees RAG Chatbot!\n\n" +
            "Ask me anything about ScriptBees' services, technologies, or projects.\n\n" +
            "Example:\n• What services does ScriptBees offer?\n• Tell me about ScriptBees' cloud expertise.",
            false
        );
    }, 200);
}

// ========== RUN ON LOAD ==========
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// ========== RUNTIME UTILITIES ==========
window.updateApiKey = function(newKey) {
    API_KEY = String(newKey || '').trim();
    console.log('🔑 API key updated:', API_KEY ? API_KEY.substring(0, 10) + '...' : '(empty)');
};

window.updateBackendUrl = function(newUrl) {
    API_URL = String(newUrl || '').trim();
    console.log('🔗 Backend URL updated:', API_URL);
};

window.clearChat = function() {
    if (chatMessages) chatMessages.innerHTML = '';
    console.log('🗑️ Chat cleared');
    // re-show welcome message
    setTimeout(() => {
        addMessage("👋 Welcome back — ask another question!", false);
    }, 150);
};

// Export small api for debugging
window.scriptbeesChat = {
    sendQuery,
    checkHealth,
    clearChat,
    updateApiKey: window.updateApiKey,
    updateBackendUrl: window.updateBackendUrl
};

console.log('🐝 ScriptBees chat utilities available via window.scriptbeesChat');
