function toggleBibtex(id) {
    var element = document.getElementById(id);
    if (element) {
        if (element.style.display === 'none' || element.style.display === '') {
            element.style.display = 'block';
        } else {
            element.style.display = 'none';
        }
    } else {
        console.error('BibTeX element not found:', id);
    }
}

