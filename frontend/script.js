class ChatApp {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.isProcessing = false;
        this.typingTimeout = null;
        this.apiBaseUrl = '/api/v1';
        this.initializeApp();
    }

    initializeElements() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.fileInput = document.getElementById('fileInput');
        this.fileUploadBtn = document.getElementById('fileUploadBtn');
        this.charCount = document.getElementById('charCount');
        this.newChatBtn = document.getElementById('newChatBtn');
    }

    attachEventListeners() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('input', () => this.handleInputChange());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.fileUploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.newChatBtn.addEventListener('click', () => this.startNewChat());

        this.messageInput.addEventListener('paste', (e) => this.handlePaste(e));
    }

    handleInputChange() {
        const text = this.messageInput.value.trim();
        const charLength = this.messageInput.value.length;
        
        this.charCount.textContent = `${charLength}/2000`;
        this.sendBtn.disabled = text.length === 0 || this.isProcessing;

        this.autoResizeTextarea();
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;

        // Store current question for display
        this.currentQuestion = message;
        
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.handleInputChange();

        try {
            this.showTypingIndicator();
            await this.sendQueryToBackend(message);
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, something went wrong. Please try again.', 'assistant');
        } finally {
            this.hideTypingIndicator();
        }
    }

    async sendQueryToBackend(query) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    max_context: 5,
                    similarity_threshold: 0.7
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.lastResponseData = data; // Store for detailed display
            this.addMessage(data.response || data.answer || 'I processed your query but couldn\'t generate a response.', 'assistant');
            
        } catch (error) {
            console.error('Backend query error:', error);
            throw error;
        }
    }

    addMessage(content, sender, isUpload = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = sender === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-wrapper';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // For assistant messages with structured content, add full system display
        if (sender === 'assistant' && this.lastResponseData) {
            const data = this.lastResponseData;
            
            // Build complete response with user question and system proof
            let fullResponse = '';
            
            // User question
            if (data.user_question) {
                fullResponse += `User: ${data.user_question}\n\n`;
            }
            
            // Main answer
            fullResponse += content;
            
            // Agent execution
            if (data.agent_execution) {
                fullResponse += '\n\n⚙️ Agent Execution:\n';
                fullResponse += `- Query Agent → ${data.agent_execution.query_agent.action}\n`;
                fullResponse += `- Retrieval Agent → ${data.agent_execution.retrieval_agent.action}\n`;
                fullResponse += `- Generation Agent → ${data.agent_execution.generation_agent.action}\n`;
                fullResponse += `- Validation Agent → ${data.agent_execution.validation_agent.action}`;
            }
            
            // Retrieved chunks
            if (data.sources && data.sources.length > 0) {
                fullResponse += '\n\n📚 Top Retrieved Chunks:\n';
                data.sources.forEach((source, index) => {
                    fullResponse += `${index + 1}. ${source.filename} (${source.chunk_id}) – ${source.similarity}\n`;
                });
            }
            
            contentDiv.innerHTML = `<p style="white-space: pre-wrap;">${this.escapeHtml(fullResponse)}</p>`;
        } else {
            contentDiv.innerHTML = `<p>${this.escapeHtml(content)}</p>`;
        }
        
        // Add timestamp
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = this.getCurrentTime();

        if (sender === 'user') {
            contentWrapper.appendChild(contentDiv);
            contentWrapper.appendChild(timestampDiv);
            messageDiv.appendChild(contentWrapper);
            messageDiv.appendChild(avatarDiv);
        } else {
            messageDiv.appendChild(avatarDiv);
            contentWrapper.appendChild(contentDiv);
            contentWrapper.appendChild(timestampDiv);
            messageDiv.appendChild(contentWrapper);
        }

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message typing-message';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-wrapper">
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingMessage = document.querySelector('.typing-message');
        if (typingMessage) {
            typingMessage.remove();
        }
    }

    async initializeApp() {
        try {
            await this.checkBackendHealth();
            // Backend connected successfully - no message shown
        } catch (error) {
            console.error('Backend health check failed:', error);
            this.addMessage('⚠️ Backend connection failed. The app will run in demo mode. Please check if the backend server is running on port 8000.', 'assistant');
        }
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (!response.ok) {
                throw new Error(`Health check failed: ${response.status}`);
            }
            const health = await response.json();
            console.log('Backend health:', health);
            return health;
        } catch (error) {
            throw error;
        }
    }

    handleFileUpload(event) {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;

        files.forEach(file => this.processFile(file));
        event.target.value = '';
    }

    async processFile(file) {
        // Create processing message
        const processingMessageId = this.addProcessingMessage(file.name, file.size);
        
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            const result = await response.json();
            
            // Update to success state and show assistant message in one function
            this.updateProcessingMessage(processingMessageId, 'success', file.name);
            
        } catch (error) {
            // Update to error state and show error message in one function
            this.updateProcessingMessage(processingMessageId, 'error');
        }
    }

    addProcessingMessage(filename, fileSize) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message user-message`;
        const messageId = `processing-${Date.now()}`;
        messageDiv.id = messageId;
        
        // Store file size for later use
        messageDiv.dataset.fileSize = fileSize;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = '<i class="fas fa-user"></i>';

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-wrapper';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <p>Processing: ${filename} (${this.formatFileSize(fileSize)})</p>
        `;

        // Add processing class to the entire message content
        contentDiv.classList.add('file-processing');

        // Add timestamp
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = this.getCurrentTime();

        contentWrapper.appendChild(contentDiv);
        contentWrapper.appendChild(timestampDiv);
        messageDiv.appendChild(contentWrapper);
        messageDiv.appendChild(avatarDiv);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageId;
    }

    updateProcessingMessage(messageId, status, filename = null) {
        const messageElement = document.getElementById(messageId);
        if (!messageElement) return;

        const contentDiv = messageElement.querySelector('.message-content');
        if (!contentDiv) return;

        // Remove existing classes
        contentDiv.classList.remove('file-processing', 'file-success', 'file-error');
        
        if (status === 'success') {
            contentDiv.classList.add('file-success');
            
            // Update the same message to show "Processed" instead of creating new message
            const fileSize = this.formatFileSize(messageElement.dataset.fileSize || 0);
            contentDiv.innerHTML = `
                <p>✅ Processed: ${filename} (${fileSize})</p>
            `;
            
            // Show assistant message after a short delay
            setTimeout(() => {
                this.showTypingIndicator();
                setTimeout(() => {
                    this.hideTypingIndicator();
                    this.addMessage(`I've processed your document "${filename}" and it's now ready for queries. You can ask me questions about its content.`, 'assistant');
                }, 1000);
            }, 500);
        } else if (status === 'error') {
            contentDiv.classList.add('file-error');
            
            // Update the same message to show "Error"
            contentDiv.innerHTML = `
                <p>❌ Upload Failed</p>
            `;
            
            // Show error message after a short delay
            setTimeout(() => {
                this.addMessage(`Sorry, I encountered an error. Please try again.`, 'assistant');
            }, 500);
        }
    }

    handlePaste(event) {
        const items = event.clipboardData.items;
        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                event.preventDefault();
                const file = item.getAsFile();
                if (file) {
                    this.processFile(file);
                }
                break;
            }
        }
    }

    startNewChat() {
        // Clear all messages except the initial greeting
        this.messagesContainer.innerHTML = `
            <div class="message assistant-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-wrapper">
                    <div class="message-content">
                        <p>Hello! I'm your AI assistant. How can I help you with your documents today?</p>
                    </div>
                    <div class="message-timestamp">${this.getCurrentTime()}</div>
                </div>
            </div>
        `;
        
        // Clear input field
        this.messageInput.value = '';
        this.handleInputChange();
        
        // Scroll to top
        this.scrollToBottom();
        
        // Focus on input
        this.messageInput.focus();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
});

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        document.title = 'Chat Application';
    } else {
        document.title = '💬 New messages';
    }
});
