// assets/socket.js
console.log('Socket.js file loaded');

// Fonction pour initialiser Socket.IO
function initializeSocket() {
    console.log('Initializing Socket.IO...');
    
    // Initialisation de Socket.IO
    const socket = io('/', {
        transports: ['websocket', 'polling'],  // Allow both WebSocket and polling
        reconnection: true,                    // Enable reconnection
        reconnectionAttempts: 5,               // Try to reconnect 5 times
        reconnectionDelay: 1000,              // Wait 1 second between attempts
        forceNew: true                          // Force a new connection
    });

    // Debug events
    socket.on('connect', () => {
        console.log('Socket connected!', socket.id);
        socket.emit('message', 'Hello from client!');
        // Debug: Check if elements exist
        const textarea = document.getElementById('terminal_output');
        const progressBar = document.getElementById('progress_bar');
    });

    socket.on('disconnect', () => {
        console.log('Socket disconnected!');
    });

    socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
    });

    // Application events
    socket.on('update_terminal', (data) => {
        console.log('Received terminal update:', data);
        const textarea = document.getElementById('terminal_output');
        if (textarea) {
            const shouldScroll = textarea.scrollTop + textarea.clientHeight === textarea.scrollHeight;
            textarea.value += data.output;
            
            // Ne faire défiler que si l'utilisateur était déjà en bas
            if (shouldScroll) {
                textarea.scrollTop = textarea.scrollHeight;
            }
        } else {
            console.error('Textarea not found!');
        }
    });

    socket.on('update_progress', (data) => {
        const progressBar = document.getElementById('progress_bar');
        if (progressBar) {
            const percentage = Math.round(data.progress); // Arrondir le pourcentage
            progressBar.style.width = `${percentage}%`;
            progressBar.textContent = `${percentage}%`;
        } else {
            console.error('Progress bar not found!');
        }
    });

    return socket;
}

// Initialize Socket.IO when the page is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSocket);
} else {
    const socket = initializeSocket();
}
