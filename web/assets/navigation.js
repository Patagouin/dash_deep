// assets/navigation.js
document.addEventListener('DOMContentLoaded', function() {
    // Récupérer le chemin actuel
    const currentPath = window.location.pathname.replace('/', '') || 'dashboard';
    
    // Récupérer tous les liens de navigation
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Fonction pour mettre à jour les classes actives
    function updateActiveLink() {
        navLinks.forEach(link => {
            const page = link.getAttribute('data-page');
            if (window.location.pathname.includes(page)) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
    
    // Mettre à jour l'état initial
    updateActiveLink();
    
    // Ajouter les écouteurs d'événements pour les clics
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Retirer la classe active de tous les liens
            navLinks.forEach(l => l.classList.remove('active'));
            // Ajouter la classe active au lien cliqué
            this.classList.add('active');
        });
    });
});
