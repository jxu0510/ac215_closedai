document.addEventListener('DOMContentLoaded', () => {
    // 动态加载导航栏或其他页面组件的逻辑
    const hamburger = document.querySelector('.hamburger');
    const mobileMenu = document.querySelector('.mobile-menu');

    hamburger.addEventListener('click', () => {
        mobileMenu.classList.toggle('active');
    });

    // Chatbot 的功能
    function addMessage(sender, text) {
        const chatWindow = document.getElementById('chat-window');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.textContent = text;
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function sendMessage() {
        const userInput = document.getElementById('user-input');
        const message = userInput.value.trim();
        if (!message) return;

        addMessage('user', message);
        userInput.value = '';

        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => addMessage('bot', data.reply))
        .catch(() => addMessage('bot', 'Error: Unable to get a response.'));
    }

    document.getElementById('user-input').addEventListener('keydown', (event) => {
        if (event.key === 'Enter') sendMessage();
    });

    document.querySelector('button').addEventListener('click', sendMessage);
});