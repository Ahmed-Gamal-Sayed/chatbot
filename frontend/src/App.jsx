import React, { useState } from "react";
import axios from "axios";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (input.trim() === "") return;

    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const res = await axios.post("http://localhost:8000/chat", {
        message: input,
      });

      const botMsg = { sender: "bot", text: res.data.response };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      const errorMsg = { sender: "bot", text: "[ERROR] Try agen..." };
      setMessages((prev) => [...prev, errorMsg]);
    }

    setInput("");
  };

  return (
    <>
      <div className="title">
        <h1>Chat BOT App</h1>
      </div>
      <div className="chat-container">
        <div className="chat-box">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              {msg.text}
            </div>
          ))}
        </div>
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </>
  );
}
