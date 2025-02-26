async function submitQuery() {
    const query = document.getElementById('query-input').value;
    const response = await fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: query }),
    });
    const data = await response.json();
    document.getElementById('answer').innerText = data.answer;
}
