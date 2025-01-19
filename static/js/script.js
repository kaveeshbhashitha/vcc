// script.js
const speakButton = document.getElementById('speak-btn');
const displayScreen = document.getElementById('display-screen');

// Initialize Speech Recognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();

recognition.lang = 'en-US';
recognition.interimResults = false;

// When the Speak button is clicked
speakButton.addEventListener('click', () => {
    recognition.start();
    speakButton.disabled = true;
    displayScreen.textContent = "Listening...";
});

// When speech is detected
recognition.onresult = (event) => {
    const spokenText = event.results[0][0].transcript; // Get the spoken text
    displayScreen.textContent = "You said: " + spokenText;

    // Send the spoken text to Flask backend
    fetch('/process-command', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ command: spokenText }),
    })
    .then(response => response.json())
    .then(data => {
        // Display the result on the calculator screen
        displayScreen.textContent = "= " + data.result;
    })
    .catch(error => {
        displayScreen.textContent = "Error: Could not process the command.";
        console.error("Error:", error);
    })
    .finally(() => {
        speakButton.disabled = false; // Re-enable the button
    });
};

// Handle end of speech detection
recognition.onspeechend = () => {
    recognition.stop();
    speakButton.disabled = false;
};

// Handle recognition errors
recognition.onerror = (event) => {
    displayScreen.textContent = "Error: " + event.error;
    speakButton.disabled = false;
};
