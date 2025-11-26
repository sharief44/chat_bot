/**
 * ScriptBees RAG Chatbot Frontend
 * Connects to POST /api/ask endpoint with API key authentication
 * Configured for ScriptBees.com content
 */

// ==========================================
// CONFIGURATION
// ==========================================

const API_URL = 'http://localhost:8000'.trim(); // Backend URL
let API_KEY = 'X2Cli1ZSPhHHAHlfZkOEPRWIqtd1TQD9ErH705-HMc4'; // Default API Key

// ==========================================
// DOM ELEMENTS
// ==========================================

const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// ==========================================
// MESSAGE FUNCTIONS
// ==========================================

function addMessage(text, isUser = false, sources = null, isError = false, metadata = null) {
    const msg = document.createElement('div');
    msg.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const content = document.createElement('div');
    content.className = 'message-content';
    if (isError) content.classList.add('error-message');

    const textElem = document.createElement('p');
    textElem.innerHTML = text.replace(/\n/g, '<br>');
    content.appendChild(textElem);

    // Add sources if available
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
            link.textContent = url.replace('https://scriptbees.com', '').replace('https://www.scriptbees.com', '') || '/home';
            link.className = 'source-link';
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            srcDiv.appendChild(link);
        });

        content.appendChild(srcDiv);
    }

    // Add metadata (response time, cached status)
    if (metadata) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'metadata';
        metaDiv.style.fontSize = '0.75rem';
        metaDiv.style.color = '#64748b';
        metaDiv.style.marginTop = '8px';
        metaDiv.style.fontStyle = 'italic';
        
        let metaText = '';
        if (metadata.response_time_seconds) {
            metaText += `⚡ ${metadata.response_time_seconds.toFixed(2)}s`;
        }
        if (metadata.cached) {
            metaText += ' • 💾 Cached';
        }
        metaDiv.textContent = metaText;
        if (metaText) content.appendChild(metaDiv);
    }

    msg.appendChild(content);
    chatMessages.appendChild(msg);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

function showTyping() {
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

// ==========================================
// API FUNCTIONS
// ==========================================

async function sendQuery(question) {
    try {
        const response = await fetch(`${API_URL}/api/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            if (response.status === 403) throw new Error('Invalid API key. Please check your configuration.');
            if (response.status === 503) throw new Error('Server still loading models. Please wait...');
            if (response.status === 404) throw new Error('API endpoint not found. Check backend is running.');

            throw new Error(err.detail || err.error || `Server error (${response.status})`);
        }

        return await response.json();

    } catch (err) {
        if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
            throw new Error(
                `Cannot connect to backend at ${API_URL}\n\n` +
                `Troubleshooting:\n` +
                `1. Ensure backend is running (python main.py)\n` +
                `2. Check CORS is enabled\n` +
                `3. Verify API key is correct\n` +
                `4. Check firewall/network settings`
            );
        }
        throw err;
    }
}

async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();

        console.log('🐝 ScriptBees RAG Server Health:', data);

        if (!data.retriever_loaded || !data.generator_loaded) {
            addMessage(
                '⚠️ Server starting up. AI models are loading...\n' +
                'This may take 1-2 minutes. Please wait.',
                false
            );
            // Retry health check in 5 seconds
            setTimeout(checkHealth, 5000);
        } else {
            console.log(`✅ Server ready with ${data.num_documents} ScriptBees documents indexed`);
        }

        return data;
    } catch (err) {
        addMessage(
            `❌ Cannot connect to backend at ${API_URL}\n\n` +
            `Please ensure:\n` +
            `1. Backend server is running\n` +
            `2. Port 8000 is accessible\n` +
            `3. CORS is configured correctly`,
            false,
            null,
            true
        );
        return null;
    }
}

// ==========================================
// FORM HANDLING
// ==========================================

async function handleSubmit(e) {
    e.preventDefault();

    const question = userInput.value.trim();
    if (!question) return;

    // Add user message
    addMessage(question, true);

    // Reset input
    userInput.value = '';
    userInput.disabled = true;
    sendButton.disabled = true;
    sendButton.textContent = 'Thinking...';

    showTyping();

    try {
        const response = await sendQuery(question);
        removeTyping();
        
        // Add bot response with metadata
        addMessage(
            response.answer,
            false,
            response.sources,
            false,
            {
                response_time_seconds: response.response_time_seconds,
                cached: response.cached
            }
        );

        console.log('Response metadata:', {
            cached: response.cached,
            time: response.response_time_seconds,
            sources: response.sources?.length || 0
        });

    } catch (err) {
        removeTyping();
        addMessage(
            `⚠️ Error:\n${err.message}`,
            false,
            null,
            true
        );
        console.error('Query error:', err);

    } finally {
        userInput.disabled = false;
        sendButton.disabled = false;
        sendButton.textContent = 'Send';
        userInput.focus();
    }
}

// ==========================================
// KEYBOARD SHORTCUTS
// ==========================================

userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
    }
});

// ==========================================
// INITIALIZATION
// ==========================================

function init() {
    console.log('🐝 ScriptBees RAG Chatbot Initialized');
    console.log(`Backend: ${API_URL}`);
    console.log(`API Key: ${API_KEY.substring(0, 8)}...`);

    // Check server health
    checkHealth();

    // Show welcome message
    setTimeout(() => {
        addMessage(
            "👋 Welcome to ScriptBees RAG Chatbot!\n\n" +
            "Ask me anything about ScriptBees' services, technologies, expertise, or projects.\n\n" +
            "Example questions:\n" +
            "• What services does ScriptBees offer?\n" +
            "• Tell me about ScriptBees' expertise in cloud solutions\n" +
            "• What industries does ScriptBees serve?\n" +
            "• What is ScriptBees' approach to quality assurance?",
            false
        );
    }, 200);
}

// ==========================================
// EVENT LISTENERS
// ==========================================

chatForm.addEventListener('submit', handleSubmit);

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

// Allow runtime API key updates (for testing)
window.updateApiKey = function(newKey) {
    API_KEY = String(newKey || '').trim();
    console.log('🔑 API key updated:', API_KEY ? API_KEY.substring(0, 10) + '...' : '(empty)');
};

// Allow runtime backend URL updates (for testing)
window.updateBackendUrl = function(newUrl) {
    API_URL = String(newUrl || '').trim();
    console.log('🔗 Backend URL updated:', API_URL);
};

// Clear chat history
window.clearChat = function() {
    chatMessages.innerHTML = '';
    console.log('🗑️ Chat cleared');
    init(); // Show welcome message again
};

// Export for debugging
window.scriptbeesChat = {
    sendQuery,
    checkHealth,
    clearChat,
    updateApiKey: window.updateApiKey,
    updateBackendUrl: window.updateBackendUrl
};

console.log('🐝 ScriptBees chat utilities available via window.scriptbeesChat');