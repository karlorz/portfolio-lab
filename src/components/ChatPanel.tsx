import React, { useState, useRef, useEffect } from 'react';
import type { ChatMessage, ChatSuggestion } from '../types/live';

interface ChatPanelProps {
  expanded?: boolean;
  onToggleExpand?: () => void;
}

const SUGGESTED_QUERIES: ChatSuggestion[] = [
  { label: 'Equity exposure', query: 'What is my equity exposure?', category: 'portfolio' },
  { label: 'Current drawdown', query: 'What is my current drawdown?', category: 'risk' },
  { label: 'Active overlays', query: 'Which overlays are currently active?', category: 'overlays' },
  { label: 'Bearish signals', query: 'Which signals are bearish right now?', category: 'signals' },
  { label: 'Trading costs', query: 'How much slippage have I had this month?', category: 'costs' },
  { label: 'Risk metrics', query: 'What are my VaR and CVaR numbers?', category: 'risk' },
];

export function ChatPanel({ expanded: expandedProp, onToggleExpand }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [internalExpanded, setInternalExpanded] = useState(expandedProp ?? false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isControlled = expandedProp !== undefined && onToggleExpand !== undefined;
  const expanded = isControlled ? expandedProp : internalExpanded;
  const headingId = 'portfolio-assistant-heading';
  const contentId = 'portfolio-assistant-content';

  const toggleExpand = () => {
    if (onToggleExpand) {
      onToggleExpand();
      return;
    }
    setInternalExpanded((current) => !current);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendQuery = async (question: string) => {
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('/api/portfolio-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      if (response.ok) {
        const data = await response.json();
        const assistantMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.answer,
          timestamp: new Date().toISOString(),
        };
        setMessages(prev => [...prev, assistantMsg]);
      } else {
        throw new Error('Query failed');
      }
    } catch {
      const errorMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Unable to reach portfolio assistant. Check that the server is running.',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !loading) {
      sendQuery(input.trim());
    }
  };

  const handleSuggestionClick = (suggestion: ChatSuggestion) => {
    if (!loading) {
      sendQuery(suggestion.query);
    }
  };

  return (
    <section
      className={`chat-panel ${expanded ? 'expanded' : 'collapsed'}`}
      aria-labelledby={headingId}
    >
      <div className="panel-header">
        <h3 id={headingId}>Portfolio Assistant</h3>
        <button
          type="button"
          className="panel-toggle"
          aria-expanded={expanded}
          aria-controls={contentId}
          aria-label={`${expanded ? 'Collapse' : 'Expand'} Portfolio Assistant`}
          onClick={toggleExpand}
        >
          <span className="expand-hint">{expanded ? 'Collapse' : 'Expand'}</span>
          <span aria-hidden="true">{expanded ? '▼' : '▶'}</span>
        </button>
      </div>

      <div
        id={contentId}
        className="chat-panel-content"
        hidden={!expanded}
      >
        <div className="chat-messages" role="log" aria-live="polite" aria-label="Portfolio assistant messages">
          {messages.length === 0 && (
            <div className="chat-welcome">
              <p>Ask me anything about your portfolio:</p>
              <div className="suggestion-chips">
                {SUGGESTED_QUERIES.map((sq) => (
                  <button
                    key={sq.query}
                    type="button"
                    className="suggestion-chip"
                    onClick={() => handleSuggestionClick(sq)}
                    disabled={loading}
                  >
                    {sq.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`chat-message ${msg.role}`}>
              <div className="message-role">
                {msg.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div className="message-content">
                {msg.content.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-message assistant">
              <div className="message-role">Assistant</div>
              <div className="message-content loading-dots" role="status" aria-live="polite">
                Loading…
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            name="question"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your portfolio…"
            aria-label="Ask the portfolio assistant"
            autoComplete="off"
            disabled={loading}
            className="chat-input"
          />
          <button type="submit" disabled={loading || !input.trim()} className="chat-send-btn">
            Send
          </button>
        </form>
      </div>
    </section>
  );
}
