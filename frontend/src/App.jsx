import { useState } from "react";
import "./App.css";

function App() {
  const API_URL = import.meta.env.VITE_API_URL;

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!question.trim() || loading) {
      return;
    }

    const currentQuestion = question.trim();

    // Add user's question to the chat immediately
    setMessages((prev) => [
      ...prev,
      {
        type: "user",
        text: currentQuestion,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: currentQuestion,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const data = await response.json();

      // Add backend response
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          text: data.answer,
          route: data.route,
        },
      ]);
    } catch (error) {
      console.error("Error:", error);

      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          text: "Sorry, something went wrong while connecting to the backend.",
          route: "error",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <div className="chat-container">

        <div className="chat-header">
          <h1>Multi-Agent RAG</h1>
          <p>Ask questions from your documents or the web</p>
        </div>

        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="welcome">
              <h2>Welcome 👋</h2>
              <p>
                Ask me anything. I can search your documents,
                the web, or directly use the LLM.
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${
                message.type === "user"
                  ? "user-message"
                  : "assistant-message"
              }`}
            >
              <div className="message-label">
                {message.type === "user" ? "You" : "Assistant"}
              </div>

              <div className="message-content">
                {message.text}
              </div>

              {message.type === "assistant" && message.route && (
                <div className="route">
                  Route: {message.route}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="message assistant-message">
              <div className="message-label">Assistant</div>
              <div className="message-content">
                Thinking...
              </div>
            </div>
          )}
        </div>

        <div className="chat-input-area">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            rows="1"
            disabled={loading}
          />

          <button
            onClick={sendMessage}
            disabled={loading || !question.trim()}
          >
            {loading ? "..." : "Send"}
          </button>
        </div>

      </div>
    </div>
  );
}

export default App;