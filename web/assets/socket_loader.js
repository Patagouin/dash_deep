// Load Socket.IO script dynamically
const script = document.createElement('script');
script.src = "https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js";
script.onload = function() {
    // Once Socket.IO is loaded, initialize our socket.js
    const socketScript = document.createElement('script');
    socketScript.src = "/assets/socket.js";
    document.head.appendChild(socketScript);
};
document.head.appendChild(script);
