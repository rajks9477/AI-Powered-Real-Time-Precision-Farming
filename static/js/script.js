// ---------- Chatbot widget ----------
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("chatbot-toggle");
  const panel = document.getElementById("chatbot-panel");
  const form = document.getElementById("chatbot-form");
  const input = document.getElementById("chatbot-input");
  const messages = document.getElementById("chatbot-messages");

  if (toggle && panel) {
    toggle.addEventListener("click", () => panel.classList.toggle("hidden"));
  }

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      appendMessage(text, "user-msg");
      input.value = "";

      try {
        const res = await fetch("/api/chatbot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });
        const data = await res.json();
        appendMessage(data.reply, "bot-msg");
      } catch (err) {
        appendMessage("Sorry, I couldn't reach the assistant right now.", "bot-msg");
      }
    });
  }

  function appendMessage(text, cls) {
    const div = document.createElement("div");
    div.className = cls;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  // ---------- Image upload preview ----------
  const fileInput = document.getElementById("image-input");
  const preview = document.getElementById("image-preview");
  const dropZone = document.getElementById("upload-drop");

  if (fileInput && preview) {
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = "block";
      };
      reader.readAsDataURL(file);
    });
  }
  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => e.preventDefault());
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event("change"));
      }
    });
  }

  // ---------- Weather widget ----------
  const weatherForm = document.getElementById("weather-form");
  const weatherResult = document.getElementById("weather-result");
  if (weatherForm) {
    weatherForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const city = document.getElementById("weather-city").value.trim();
      if (!city) return;
      weatherResult.innerHTML = "Loading...";
      try {
        const res = await fetch(`/api/weather?city=${encodeURIComponent(city)}`);
        const data = await res.json();
        if (data.error) {
          weatherResult.innerHTML = `<span class="flash flash-error">${data.error}</span>`;
        } else {
          weatherResult.innerHTML = `
            <div class="weather-widget">
              <div class="weather-temp">${Math.round(data.temperature)}°C</div>
              <div>
                <strong>${data.city}</strong><br>
                <span>${data.description}</span><br>
                <span class="mono">Humidity ${data.humidity}% · Wind ${data.wind_speed} m/s</span>
              </div>
            </div>`;
        }
      } catch (err) {
        weatherResult.innerHTML = `<span class="flash flash-error">Could not fetch weather.</span>`;
      }
    });
  }
});
